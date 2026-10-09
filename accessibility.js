// Accessibility HUD - Contrast Scanner & WCAG Compliance Checker
// Analyzes images for color contrast ratios per WCAG 2.1 guidelines

// WCAG 2.1 Contrast Ratio Thresholds
const WCAG_THRESHOLDS = {
    AA_NORMAL: 4.5,      // Normal text (AA)
    AA_LARGE: 3.0,       // Large text 18pt+ or 14pt+ bold (AA)
    AAA_NORMAL: 7.0,     // Normal text (AAA)
    AAA_LARGE: 4.5       // Large text (AAA)
};

// Relative luminance calculation per WCAG (sRGB)
function getRelativeLuminance(r, g, b) {
    // Convert sRGB to linear RGB
    const srgbToLinear = (c) => {
        const cNorm = c / 255;
        return cNorm <= 0.03928 
            ? cNorm / 12.92 
            : Math.pow((cNorm + 0.055) / 1.055, 2.4);
    };
    
    const rLin = srgbToLinear(r);
    const gLin = srgbToLinear(g);
    const bLin = srgbToLinear(b);
    
    // WCAG luminance coefficients
    return 0.2126 * rLin + 0.7152 * gLin + 0.0722 * bLin;
}

// Contrast ratio between two colors (WCAG formula)
function getContrastRatio(r1, g1, b1, r2, g2, b2) {
    const l1 = getRelativeLuminance(r1, g1, b1);
    const l2 = getRelativeLuminance(r2, g2, b2);
    
    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);
    
    return (lighter + 0.05) / (darker + 0.05);
}

// Get compliance level for a contrast ratio
function getComplianceLevel(ratio, isLargeText = false) {
    const normalThreshold = isLargeText ? WCAG_THRESHOLDS.AA_LARGE : WCAG_THRESHOLDS.AA_NORMAL;
    const aaaThreshold = isLargeText ? WCAG_THRESHOLDS.AAA_LARGE : WCAG_THRESHOLDS.AAA_NORMAL;
    
    if (ratio >= aaaThreshold) return 'AAA';
    if (ratio >= normalThreshold) return 'AA';
    return 'FAIL';
}

// Sample colors from image at regular intervals to find text/background pairs
function sampleImageColors(imageData, sampleStep = 4) {
    const { data, width, height } = imageData;
    const colors = new Map(); // color string -> count
    
    for (let y = 0; y < height; y += sampleStep) {
        for (let x = 0; x < width; x += sampleStep) {
            const idx = (y * width + x) * 4;
            const r = data[idx];
            const g = data[idx + 1];
            const b = data[idx + 2];
            const a = data[idx + 3];
            
            if (a < 128) continue; // Skip transparent
            
            const key = `${r},${g},${b}`;
            colors.set(key, (colors.get(key) || 0) + 1);
        }
    }
    
    // Return top N most frequent colors
    return Array.from(colors.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .map(([colorStr]) => colorStr.split(',').map(Number));
}

// Find edge pixels (potential text boundaries) using simple gradient
function findEdgePixels(imageData, threshold = 30) {
    const { data, width, height } = imageData;
    const edges = [];
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = (y * width + x) * 4;
            const idxR = (y * width + (x + 1)) * 4;
            const idxB = ((y + 1) * width + x) * 4;
            
            // Simple gradient magnitude
            const dr = Math.abs(data[idx] - data[idxR]) + Math.abs(data[idx] - data[idxB]);
            const dg = Math.abs(data[idx + 1] - data[idxR + 1]) + Math.abs(data[idx + 1] - data[idxB + 1]);
            const db = Math.abs(data[idx + 2] - data[idxR + 2]) + Math.abs(data[idx + 2] - data[idxB + 2]);
            
            const gradient = (dr + dg + db) / 3;
            
            if (gradient > threshold) {
                edges.push({ x, y, idx, gradient });
            }
        }
    }
    
    return edges;
}

// Analyze contrast at edge boundaries (text-like regions)
function analyzeEdgeContrast(imageData, edges, maxSamples = 100) {
    const { data, width } = imageData;
    const results = [];
    
    // Sample edge pixels
    const step = Math.max(1, Math.floor(edges.length / maxSamples));
    
    for (let i = 0; i < edges.length; i += step) {
        const edge = edges[i];
        const { x, y, idx } = edge;
        
        // Sample color on one side of edge
        const r1 = data[idx];
        const g1 = data[idx + 1];
        const b1 = data[idx + 2];
        
        // Sample neighbor (other side of edge)
        const neighborIdx = ((y + 1) * width + x) * 4;
        if (neighborIdx >= data.length) continue;
        
        const r2 = data[neighborIdx];
        const g2 = data[neighborIdx + 1];
        const b2 = data[neighborIdx + 2];
        
        const ratio = getContrastRatio(r1, g1, b1, r2, g2, b2);
        const compliance = getComplianceLevel(ratio);
        
        results.push({
            x, y,
            fg: { r: r1, g: g1, b: b1 },
            bg: { r: r2, g: g2, b: b2 },
            ratio: Math.round(ratio * 100) / 100,
            compliance,
            gradient: edge.gradient
        });
    }
    
    return results;
}

// Main contrast analysis function
export function analyzeContrast(imageData, options = {}) {
    const {
        sampleStep = 4,
        edgeThreshold = 30,
        maxEdgeSamples = 200,
        includeColorPairs = true
    } = options;
    
    // Find dominant colors
    const dominantColors = sampleImageColors(imageData, sampleStep);
    
    // Find edges (text boundaries)
    const edges = findEdgePixels(imageData, edgeThreshold);
    
    // Analyze contrast at edges
    const edgeResults = analyzeEdgeContrast(imageData, edges, maxEdgeSamples);
    
    // Also check dominant color pairs
    let colorPairResults = [];
    if (includeColorPairs && dominantColors.length >= 2) {
        for (let i = 0; i < dominantColors.length; i++) {
            for (let j = i + 1; j < dominantColors.length; j++) {
                const [r1, g1, b1] = dominantColors[i];
                const [r2, g2, b2] = dominantColors[j];
                const ratio = getContrastRatio(r1, g1, b1, r2, g2, b2);
                const compliance = getComplianceLevel(ratio);
                
                colorPairResults.push({
                    fg: { r: r1, g: g1, b: b1 },
                    bg: { r: r2, g: g2, b: b2 },
                    ratio: Math.round(ratio * 100) / 100,
                    compliance
                });
            }
        }
    }
    
    // Combine and sort by ratio (worst first)
    const allResults = [...edgeResults, ...colorPairResults]
        .sort((a, b) => a.ratio - b.ratio);
    
    // Summary statistics
    const total = allResults.length;
    const aaPass = allResults.filter(r => r.compliance !== 'FAIL').length;
    const aaaPass = allResults.filter(r => r.compliance === 'AAA').length;
    
    return {
        summary: {
            totalPairs: total,
            aaPass,
            aaaPass,
            aaFail: total - aaPass,
            worstRatio: allResults[0]?.ratio || 0,
            bestRatio: allResults[allResults.length - 1]?.ratio || 0
        },
        details: allResults.slice(0, 50), // Limit results
        dominantColors: dominantColors.slice(0, 10).map(([r, g, b]) => ({ r, g, b }))
    };
}

// Generate heatmap visualization of contrast issues
export function generateContrastHeatmap(imageData, analysis, options = {}) {
    const { data, width, height } = imageData;
    const outputData = new Uint8ClampedArray(data);
    
    const {
        showFailingOnly = true,
        opacity = 180
    } = options;
    
    // Create a grid of contrast results
    const gridSize = 8;
    const cellW = Math.ceil(width / gridSize);
    const cellH = Math.ceil(height / gridSize);
    
    // For each detail result, mark the region
    analysis.details.forEach(detail => {
        if (showFailingOnly && detail.compliance !== 'FAIL') return;
        
        const gx = Math.floor(detail.x / cellW);
        const gy = Math.floor(detail.y / cellH);
        
        // Color code by compliance
        let overlayR = 255, overlayG = 0, overlayB = 0; // Red for FAIL
        if (detail.compliance === 'AA') { overlayR = 255; overlayG = 165; overlayB = 0; } // Orange
        if (detail.compliance === 'AAA') { overlayR = 0; overlayG = 255; overlayB = 0; } // Green
        
        // Apply overlay to cell region
        const startX = gx * cellW;
        const startY = gy * cellH;
        const endX = Math.min(startX + cellW, width);
        const endY = Math.min(startY + cellH, height);
        
        for (let y = startY; y < endY; y++) {
            for (let x = startX; x < endX; x++) {
                const idx = (y * width + x) * 4;
                outputData[idx] = Math.round((outputData[idx] * (255 - opacity) + overlayR * opacity) / 255);
                outputData[idx + 1] = Math.round((outputData[idx + 1] * (255 - opacity) + overlayG * opacity) / 255);
                outputData[idx + 2] = Math.round((outputData[idx + 2] * (255 - opacity) + overlayB * opacity) / 255);
            }
        }
    });
    
    return new ImageData(outputData, width, height);
}

// Color blindness safe palette generator (async version for ES modules)
export async function generateSafePalette(baseColor, mode) {
    // Simulate the base color for the given CVD mode
    const { simulateColorblindness } = await import('./colorMatrices.js');
    
    const simulated = simulateColorblindness(baseColor.r, baseColor.g, baseColor.b, mode, 1.0);
    
    // Generate variations with good contrast
    const variations = [];
    const luminances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    
    luminances.forEach(l => {
        // Create color with target luminance
        const factor = l / getRelativeLuminance(simulated.r, simulated.g, simulated.b);
        variations.push({
            r: Math.min(255, Math.max(0, Math.round(simulated.r * factor))),
            g: Math.min(255, Math.max(0, Math.round(simulated.g * factor))),
            b: Math.min(255, Math.max(0, Math.round(simulated.b * factor))),
            luminance: l
        });
    });
    
    return variations;
}

// Export utilities
export {
    getContrastRatio,
    getRelativeLuminance,
    getComplianceLevel,
    WCAG_THRESHOLDS
};