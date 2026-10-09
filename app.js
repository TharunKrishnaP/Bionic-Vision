// Main Application - Webcam Pipeline, Canvas Rendering, UI Logic
import {
    SIMULATION_MATRICES,
    DALTONIZE_MATRICES,
    RGB_TO_LMS,
    LMS_TO_RGB,
    simulateColorblindness,
    daltonize,
    createCombinedMatrix
} from './colorMatrices.js';

import {
    VERTEX_SHADER_SOURCE,
    SIMULATION_FRAGMENT_SHADER_SOURCE,
    DALTONIZE_FRAGMENT_SHADER_SOURCE,
    PATHOLOGY_FRAGMENT_SHADER_SOURCE,
    SPLIT_VIEW_FRAGMENT_SHADER_SOURCE,
    ShaderProgram
} from './shaders.js';

import {
    analyzeContrast,
    generateContrastHeatmap,
    getContrastRatio,
    getComplianceLevel,
    WCAG_THRESHOLDS
} from './accessibility.js';

// DOM Elements
const canvas = document.getElementById('canvas');
const offscreenCanvas = document.getElementById('offscreenCanvas');
const ctx = canvas.getContext('2d');
const offscreenCtx = offscreenCanvas.getContext('2d');
const modeSelect = document.getElementById('mode');
const intensitySlider = document.getElementById('intensity');
const intensityValue = document.getElementById('intensityValue');
const toggleCameraBtn = document.getElementById('toggleCamera');
const uploadBtn = document.getElementById('uploadBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const exportBtn = document.getElementById('exportBtn');
const fileInput = document.getElementById('fileInput');
const daltonizationCheckbox = document.getElementById('daltonization');
const splitViewCheckbox = document.getElementById('splitView');
const splitSlider = document.getElementById('splitSlider');
const daltonizationGroup = document.getElementById('daltonizationGroup');
const splitSliderGroup = document.getElementById('splitSliderGroup');
const loading = document.getElementById('loading');
const conditionInfo = document.getElementById('conditionInfo');
const viewerContainer = document.querySelector('.viewer-container');
const accessibilityPanel = document.getElementById('accessibilityPanel');
const accessibilityContent = document.getElementById('accessibilityContent');
const analysisStatus = document.getElementById('analysisStatus');

// WebGL Context
let gl = null;
let simulationProgram = null;
let daltonizeProgram = null;
let pathologyProgram = null;
let splitViewProgram = null;
let texture = null;
let quadBuffer = null;
let texCoordBuffer = null;

// State
let videoStream = null;
let videoElement = null;
let currentMode = 'normal';
let currentIntensity = 1.0;
let isCameraOn = false;
let useDaltonization = false;
let useSplitView = false;
let splitPosition = 0.5;
let currentImage = null;
let animationFrameId = null;
let splitHandle = null;
let isDraggingSplit = false;

// Accessibility state
let currentAnalysis = null;
let isAnalyzing = false;

// Condition Information
const CONDITION_INFO = {
    normal: { name: 'Normal Vision', description: 'Standard trichromatic color vision with three functioning cone types (L, M, S).', prevalence: '~90% of population' },
    protanopia: { name: 'Protanopia (Red Blind)', description: 'Complete absence of L-cones (red). Reds appear dark/gray, confusion between red-green-brown.', prevalence: '~1% of males' },
    protanomaly: { name: 'Protanomaly (Red Weak)', description: 'Weakened L-cone sensitivity. Reduced red discrimination, milder than protanopia.', prevalence: '~1% of males' },
    deuteranopia: { name: 'Deuteranopia (Green Blind)', description: 'Complete absence of M-cones (green). Greens appear beige, confusion between green-red-brown.', prevalence: '~1% of males' },
    deuteranomaly: { name: 'Deuteranomaly (Green Weak)', description: 'Weakened M-cone sensitivity. Most common CVD, reduced green discrimination.', prevalence: '~5% of males' },
    tritanopia: { name: 'Tritanopia (Blue Blind)', description: 'Complete absence of S-cones (blue). Blues appear green, yellows appear pink/red.', prevalence: '~0.01% (rare)' },
    tritanomaly: { name: 'Tritanomaly (Blue Weak)', description: 'Weakened S-cone sensitivity. Reduced blue-yellow discrimination.', prevalence: '~0.01% (rare)' },
    achromatopsia: { name: 'Achromatopsia (Total Color Blind)', description: 'No functioning cones. Vision is grayscale with severe light sensitivity.', prevalence: '~1 in 30,000' },
    achromatomaly: { name: 'Achromatomaly (Weak Total Color)', description: 'Severely impaired cone function. Near-grayscale with minimal color perception.', prevalence: 'Very rare' },
    cataracts: { name: 'Cataracts', description: 'Clouding of the eye\'s lens causing blurred vision, glare sensitivity, and reduced contrast.', prevalence: 'Common in 60+' },
    glaucoma: { name: 'Glaucoma (Tunnel Vision)', description: 'Optic nerve damage causing peripheral vision loss, creating tunnel vision effect.', prevalence: '~3% of 40+' },
    'macular-degeneration': { name: 'Macular Degeneration', description: 'Central vision loss with blurred/distorted center, peripheral vision preserved.', prevalence: 'Common in 60+' }
};

// Initialize WebGL
function initWebGL() {
    gl = canvas.getContext('webgl', { preserveDrawingBuffer: true, alpha: false });
    if (!gl) {
        console.warn('WebGL not supported, falling back to Canvas 2D');
        return false;
    }

    // Create shader programs
    simulationProgram = new ShaderProgram(gl, VERTEX_SHADER_SOURCE, SIMULATION_FRAGMENT_SHADER_SOURCE);
    daltonizeProgram = new ShaderProgram(gl, VERTEX_SHADER_SOURCE, DALTONIZE_FRAGMENT_SHADER_SOURCE);
    pathologyProgram = new ShaderProgram(gl, VERTEX_SHADER_SOURCE, PATHOLOGY_FRAGMENT_SHADER_SOURCE);
    splitViewProgram = new ShaderProgram(gl, VERTEX_SHADER_SOURCE, SPLIT_VIEW_FRAGMENT_SHADER_SOURCE);

    if (!simulationProgram.program || !daltonizeProgram.program || !pathologyProgram.program || !splitViewProgram.program) {
        console.error('Failed to create shader programs');
        return false;
    }

    // Create quad geometry (two triangles covering -1 to 1)
    const positions = new Float32Array([
        -1, -1,  1, -1,  -1,  1,
        -1,  1,  1, -1,   1,  1
    ]);
    const texCoords = new Float32Array([
        0, 1,  1, 1,  0, 0,
        0, 0,  1, 1,  1, 0
    ]);

    quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);

    // Create texture
    texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Set viewport
    gl.viewport(0, 0, canvas.width, canvas.height);

    return true;
}

// Resize canvas to match display size
function resizeCanvas() {
    const container = viewerContainer;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    
    offscreenCanvas.width = canvas.width;
    offscreenCanvas.height = canvas.height;
    
    if (gl) {
        gl.viewport(0, 0, canvas.width, canvas.height);
    }
}

// Update texture with video frame or image
function updateTexture(source) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
}

// Render using WebGL
function renderWebGL() {
    if (!gl || !currentImage) return;

    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    updateTexture(currentImage);

    const simMatrix = SIMULATION_MATRICES[currentMode];
    const isPathology = ['cataracts', 'glaucoma', 'macular-degeneration'].includes(currentMode);
    const isNormal = currentMode === 'normal';
    const useDalton = useDaltonization && DALTONIZE_MATRICES[currentMode] && !isNormal && !isPathology;

    let program;
    if (useSplitView && !isNormal) {
        program = splitViewProgram;
    } else if (isPathology) {
        program = pathologyProgram;
    } else if (useDalton && simMatrix) {
        program = daltonizeProgram;
    } else if (simMatrix) {
        program = simulationProgram;
    } else {
        program = simulationProgram; // Normal - just passthrough
    }

    program.use();

    // Set up attributes
    const posLoc = program.getAttribLocation('a_position');
    const texLoc = program.getAttribLocation('a_texCoord');
    
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    
    gl.enableVertexAttribArray(texLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    gl.uniform1i(program.getUniformLocation('u_image'), 0);
    gl.uniform1f(program.getUniformLocation('u_intensity'), currentIntensity);

    if (program === simulationProgram || program === splitViewProgram) {
        program.setUniformMatrix3fv('u_rgbToLms', RGB_TO_LMS);
        program.setUniformMatrix3fv('u_lmsToRgb', LMS_TO_RGB);
    }

    if (program === simulationProgram) {
        gl.uniform1i(program.getUniformLocation('u_enabled'), simMatrix ? 1 : 0);
        if (simMatrix) {
            program.setUniformMatrix3fv('u_simMatrix', simMatrix);
        }
    } else if (program === daltonizeProgram) {
        const daltonMatrix = DALTONIZE_MATRICES[currentMode];
        gl.uniform1i(program.getUniformLocation('u_enabled'), daltonMatrix ? 1 : 0);
        if (daltonMatrix) {
            program.setUniformMatrix3fv('u_daltonizeMatrix', daltonMatrix);
        }
    } else if (program === pathologyProgram) {
        const conditionMap = { 'cataracts': 1, 'glaucoma': 2, 'macular-degeneration': 3 };
        gl.uniform1i(program.getUniformLocation('u_condition'), conditionMap[currentMode] || 0);
        program.setUniform2f(program.getUniformLocation('u_resolution'), canvas.width, canvas.height);
    } else if (program === splitViewProgram) {
        const leftSim = currentMode;
        const rightSim = useDaltonization ? currentMode : 'normal';
        
        gl.uniform1f(program.getUniformLocation('u_splitPos'), splitPosition);
        gl.uniform1i(program.getUniformLocation('u_useLeftSim'), simMatrix ? 1 : 0);
        gl.uniform1i(program.getUniformLocation('u_useRightSim'), useDaltonization && DALTONIZE_MATRICES[currentMode] ? 1 : 0);
        gl.uniform1i(program.getUniformLocation('u_useLeftDalton'), false);
        gl.uniform1i(program.getUniformLocation('u_useRightDalton'), useDaltonization && DALTONIZE_MATRICES[currentMode] ? 1 : 0);
        
        if (simMatrix) {
            program.setUniformMatrix3fv('u_simMatrixLeft', simMatrix);
        }
        if (useDaltonization && DALTONIZE_MATRICES[currentMode]) {
            program.setUniformMatrix3fv('u_daltonizeMatrixRight', DALTONIZE_MATRICES[currentMode]);
        }
    }

    gl.drawArrays(gl.TRIANGLES, 0, 6);
}

// Fallback Canvas 2D rendering (CPU)
function renderCanvas2D() {
    if (!currentImage) return;

    const w = offscreenCanvas.width;
    const h = offscreenCanvas.height;

    // Draw source to offscreen
    offscreenCtx.drawImage(currentImage, 0, 0, w, h);
    const imageData = offscreenCtx.getImageData(0, 0, w, h);
    const data = imageData.data;

    const simMatrix = SIMULATION_MATRICES[currentMode];
    const isPathology = ['cataracts', 'glaucoma', 'macular-degeneration'].includes(currentMode);
    const isNormal = currentMode === 'normal';

    if (useSplitView && !isNormal) {
        const splitX = Math.floor(w * splitPosition);
        
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];

                let resultR = r, resultG = g, resultB = b;

                if (x < splitX) {
                    // Left side - simulation
                    if (simMatrix && !isPathology) {
                        const sim = simulateColorblindness(r, g, b, currentMode, currentIntensity);
                        resultR = sim.r; resultG = sim.g; resultB = sim.b;
                    } else if (isPathology) {
                        // Pathology on left side
                        const sim = applyPathologyCPU(r, g, b, x, y, w, h, currentMode, currentIntensity);
                        resultR = sim.r; resultG = sim.g; resultB = sim.b;
                    }
                } else {
                    // Right side - daltonization or normal
                    if (useDaltonization && DALTONIZE_MATRICES[currentMode] && !isPathology) {
                        // First simulate, then daltonize
                        const sim = simulateColorblindness(r, g, b, currentMode, currentIntensity);
                        const dalt = daltonize(sim.r, sim.g, sim.b, currentMode, currentIntensity);
                        resultR = dalt.r; resultG = dalt.g; resultB = dalt.b;
                    }
                }

                data[idx] = resultR;
                data[idx + 1] = resultG;
                data[idx + 2] = resultB;
            }
        }
    } else {
        // Full frame processing
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];

            let resultR = r, resultG = g, resultB = b;

            if (!isNormal) {
                if (isPathology) {
                    const x = (i / 4) % w;
                    const y = Math.floor((i / 4) / w);
                    const sim = applyPathologyCPU(r, g, b, x, y, w, h, currentMode, currentIntensity);
                    resultR = sim.r; resultG = sim.g; resultB = sim.b;
                } else if (simMatrix) {
                    const sim = simulateColorblindness(r, g, b, currentMode, currentIntensity);
                    resultR = sim.r; resultG = sim.g; resultB = sim.b;

                    if (useDaltonization && DALTONIZE_MATRICES[currentMode]) {
                        const dalt = daltonize(resultR, resultG, resultB, currentMode, currentIntensity);
                        resultR = dalt.r; resultG = dalt.g; resultB = dalt.b;
                    }
                }
            }

            data[i] = resultR;
            data[i + 1] = resultG;
            data[i + 2] = resultB;
        }
    }

    ctx.putImageData(imageData, 0, 0);

    // Draw split line if in split view
    if (useSplitView && !isNormal) {
        drawSplitLine();
    }
}

function applyPathologyCPU(r, g, b, x, y, w, h, mode, intensity) {
    const centerX = w / 2;
    const centerY = h / 2;
    const dx = (x - centerX) / (w / 2);
    const dy = (y - centerY) / (h / 2);
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (mode === 'cataracts') {
        // Simple blur approximation + contrast reduction
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        const factor = intensity * 0.5;
        return {
            r: r * (1 - factor) + gray * factor,
            g: g * (1 - factor) + gray * factor,
            b: b * (1 - factor) + gray * factor
        };
    } else if (mode === 'glaucoma') {
        const innerRadius = 0.3 + (1 - intensity) * 0.4;
        const outerRadius = 0.7 + (1 - intensity) * 0.3;
        let vignette = 0;
        if (dist > innerRadius) {
            vignette = Math.min(1, (dist - innerRadius) / (outerRadius - innerRadius));
        }
        vignette *= intensity;
        return {
            r: r * (1 - vignette),
            g: g * (1 - vignette),
            b: b * (1 - vignette)
        };
    } else if (mode === 'macular-degeneration') {
        const blurRadius = intensity * 0.15;
        if (dist < blurRadius) {
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;
            const factor = (blurRadius - dist) / blurRadius * intensity * 0.5;
            return {
                r: r * (1 - factor) + gray * factor,
                g: g * (1 - factor) + gray * factor,
                b: b * (1 - factor) + gray * factor
            };
        }
    }
    return { r, g, b };
}

function drawSplitLine() {
    const x = canvas.width * splitPosition;
    ctx.save();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.shadowColor = '#fff';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
    
    // Draw handle
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#667eea';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(x, canvas.height / 2, 18, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    
    // Handle lines
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - 10, canvas.height / 2 - 15);
    ctx.lineTo(x - 10, canvas.height / 2 + 15);
    ctx.moveTo(x + 10, canvas.height / 2 - 15);
    ctx.lineTo(x + 10, canvas.height / 2 + 15);
    ctx.stroke();
    ctx.restore();
}

// Animation loop
function animate() {
    if (videoElement && videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        currentImage = videoElement;
    }
    
    if (currentImage) {
        if (gl && simulationProgram.program) {
            renderWebGL();
        } else {
            renderCanvas2D();
        }
    }
    
    animationFrameId = requestAnimationFrame(animate);
}

// Webcam handling
async function startCamera() {
    try {
        loading.classList.add('active');
        console.log('Requesting camera access...');
        
        // Try back camera first, fallback to any camera
        let constraints = {
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        };
        
        try {
            videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (err) {
            console.log('Back camera failed, trying any camera:', err.message);
            constraints = { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };
            videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        }

        videoElement = document.createElement('video');
        videoElement.srcObject = videoStream;
        videoElement.playsInline = true; // Important for iOS Safari
        videoElement.muted = true;
        
        await videoElement.play().catch(err => {
            console.error('Video play failed:', err);
            throw err;
        });

        await new Promise((resolve, reject) => {
            videoElement.onloadeddata = resolve;
            videoElement.onerror = reject;
            setTimeout(() => reject(new Error('Video load timeout')), 10000);
        });

        console.log('Camera started successfully', videoElement.videoWidth, 'x', videoElement.videoHeight);
        
        toggleCameraBtn.textContent = 'Stop Webcam';
        toggleCameraBtn.classList.add('active');
        isCameraOn = true;
        uploadBtn.disabled = true;
        
        resizeCanvas();
        animate();
        
        loading.classList.remove('active');
    } catch (err) {
        console.error('Camera error:', err);
        alert('Unable to access camera: ' + err.message + '\n\nTry: 1) Allow camera permission 2) Close other apps using camera 3) Use Upload Image instead');
        loading.classList.remove('active');
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    if (videoElement) {
        videoElement.srcObject = null;
        videoElement = null;
    }
    currentImage = null;
    toggleCameraBtn.textContent = 'Start Webcam';
    toggleCameraBtn.classList.remove('active');
    isCameraOn = false;
    uploadBtn.disabled = false;
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Image upload handling
function handleFileUpload(file) {
    if (!file || !file.type.startsWith('image/')) return;

    loading.classList.add('active');
    
    const img = new Image();
    img.onload = () => {
        currentImage = img;
        stopCamera();
        resizeCanvas();
        
        // Draw initial frame
        if (gl && simulationProgram.program) {
            renderWebGL();
        } else {
            renderCanvas2D();
        }
        
        loading.classList.remove('active');
    };
    img.onerror = () => {
        alert('Failed to load image');
        loading.classList.remove('active');
    };
    img.src = URL.createObjectURL(file);
}

// Update UI for current mode
function updateModeUI() {
    const isColorBlind = ['protanopia', 'protanomaly', 'deuteranopia', 'deuteranomaly', 'tritanopia', 'tritanomaly', 'achromatopsia', 'achromatomaly'].includes(currentMode);
    const isPathology = ['cataracts', 'glaucoma', 'macular-degeneration'].includes(currentMode);
    const isNormal = currentMode === 'normal';

    // Show/hide daltonization for color blindness only
    daltonizationGroup.style.display = (isColorBlind && !isNormal) ? 'flex' : 'none';
    
    // Show/hide split slider
    splitSliderGroup.style.display = useSplitView && !isNormal ? 'block' : 'none';

    // Update condition info
    const info = CONDITION_INFO[currentMode];
    if (info) {
        conditionInfo.innerHTML = `
            <h4>${info.name}</h4>
            <p>${info.description}</p>
            <p><strong>Prevalence:</strong> ${info.prevalence}</p>
        `;
    }

    // Disable daltonization for pathologies
    if (isPathology) {
        daltonizationCheckbox.checked = false;
        useDaltonization = false;
    }
}

// Event Listeners
modeSelect.addEventListener('change', (e) => {
    currentMode = e.target.value;
    updateModeUI();
    
    if (currentImage) {
        if (gl && simulationProgram.program) {
            renderWebGL();
        } else {
            renderCanvas2D();
        }
    }
});

intensitySlider.addEventListener('input', (e) => {
    currentIntensity = e.target.value / 100;
    intensityValue.textContent = e.target.value + '%';
    
    if (currentImage) {
        if (gl && simulationProgram.program) {
            renderWebGL();
        } else {
            renderCanvas2D();
        }
    }
});

daltonizationCheckbox.addEventListener('change', (e) => {
    useDaltonization = e.target.checked;
    updateModeUI();
    
    if (currentImage) {
        if (gl && simulationProgram.program) {
            renderWebGL();
        } else {
            renderCanvas2D();
        }
    }
});

splitViewCheckbox.addEventListener('change', (e) => {
    useSplitView = e.target.checked;
    updateModeUI();
    
    if (currentImage) {
        if (gl && simulationProgram.program) {
            renderWebGL();
        } else {
            renderCanvas2D();
        }
    }
});

splitSlider.addEventListener('input', (e) => {
    splitPosition = e.target.value / 100;
    
    if (currentImage && useSplitView) {
        if (gl && simulationProgram.program) {
            renderWebGL();
        } else {
            renderCanvas2D();
        }
    }
});

toggleCameraBtn.addEventListener('click', () => {
    if (isCameraOn) {
        stopCamera();
    } else {
        startCamera();
    }
});

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
    }
});

analyzeBtn.addEventListener('click', analyzeCurrentFrame);

exportBtn.addEventListener('click', exportCurrentFrame);

// Split line dragging - mouse and touch support
function getClientX(e) {
    return e.touches ? e.touches[0].clientX : e.clientX;
}

function startDrag(e) {
    if (!useSplitView || currentMode === 'normal') return;
    
    const rect = canvas.getBoundingClientRect();
    const x = getClientX(e) - rect.left;
    const splitX = canvas.width * splitPosition / (canvas.width / rect.width);
    
    if (Math.abs(x - splitX) < 30) { // Larger hit area for touch
        isDraggingSplit = true;
        canvas.style.cursor = 'ew-resize';
        e.preventDefault(); // Prevent scrolling
    }
}

function handleDrag(e) {
    if (!isDraggingSplit) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = getClientX(e) - rect.left;
    splitPosition = Math.max(0.1, Math.min(0.9, x / rect.width));
    splitSlider.value = Math.round(splitPosition * 100);
    
    if (currentImage) {
        if (gl && simulationProgram.program) {
            renderWebGL();
        } else {
            renderCanvas2D();
        }
    }
    e.preventDefault();
}

function endDrag() {
    isDraggingSplit = false;
    canvas.style.cursor = 'default';
}

// Mouse events
canvas.addEventListener('mousedown', startDrag);
window.addEventListener('mousemove', handleDrag);
window.addEventListener('mouseup', endDrag);

// Touch events
canvas.addEventListener('touchstart', startDrag, { passive: false });
window.addEventListener('touchmove', handleDrag, { passive: false });
window.addEventListener('touchend', endDrag);

// Window resize
window.addEventListener('resize', () => {
    resizeCanvas();
    if (currentImage) {
        if (gl && simulationProgram.program) {
            renderWebGL();
        } else {
            renderCanvas2D();
        }
    }
});

// Contrast Analysis
async function analyzeCurrentFrame() {
    if (!currentImage || isAnalyzing) return;
    
    isAnalyzing = true;
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    accessibilityPanel.style.display = 'block';
    analysisStatus.textContent = 'Scanning...';
    analysisStatus.className = 'status-badge pending';
    accessibilityContent.innerHTML = '<p>Analyzing contrast ratios...</p>';
    
    // Small delay to let UI update
    await new Promise(r => setTimeout(r, 50));
    
    try {
        // Get current frame data
        let imageData;
        if (gl && simulationProgram.program) {
            // Read from WebGL canvas
            const pixels = new Uint8Array(canvas.width * canvas.height * 4);
            gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
            // WebGL reads bottom-up, need to flip
            const flipped = new Uint8ClampedArray(pixels.length);
            for (let y = 0; y < canvas.height; y++) {
                const srcRow = (canvas.height - 1 - y) * canvas.width * 4;
                const dstRow = y * canvas.width * 4;
                flipped.set(pixels.subarray(srcRow, srcRow + canvas.width * 4), dstRow);
            }
            imageData = new ImageData(flipped, canvas.width, canvas.height);
        } else {
            // Use offscreen canvas
            offscreenCtx.drawImage(currentImage, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
            imageData = offscreenCtx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
        }
        
        // Run analysis
        currentAnalysis = analyzeContrast(imageData, {
            sampleStep: 4,
            edgeThreshold: 25,
            maxEdgeSamples: 300
        });
        
        // Display results
        displayAnalysisResults(currentAnalysis);
        
    } catch (err) {
        console.error('Analysis error:', err);
        accessibilityContent.innerHTML = `<p style="color: #dc3545;">Analysis failed: ${err.message}</p>`;
        analysisStatus.textContent = 'Error';
        analysisStatus.className = 'status-badge fail';
    } finally {
        isAnalyzing = false;
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Contrast';
    }
}

function displayAnalysisResults(analysis) {
    const { summary, details, dominantColors } = analysis;
    
    // Determine overall status
    let overallStatus = 'fail';
    let statusText = 'FAIL';
    if (summary.aaaPass > summary.totalPairs * 0.5) {
        overallStatus = 'pass-aaa';
        statusText = 'AAA PASS';
    } else if (summary.aaPass > summary.totalPairs * 0.5) {
        overallStatus = 'pass-aa';
        statusText = 'AA PASS';
    }
    
    analysisStatus.textContent = statusText;
    analysisStatus.className = `status-badge ${overallStatus}`;
    
    // Build HTML
    let html = `
        <div class="accessibility-summary">
            <div class="summary-item aaa">
                <div class="summary-value">${summary.aaaPass}</div>
                <div class="summary-label">WCAG AAA</div>
            </div>
            <div class="summary-item aa">
                <div class="summary-value">${summary.aaPass}</div>
                <div class="summary-label">WCAG AA</div>
            </div>
            <div class="summary-item fail">
                <div class="summary-value">${summary.aaFail}</div>
                <div class="summary-label">Below AA</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">${summary.worstRatio}:1</div>
                <div class="summary-label">Worst Ratio</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">${summary.bestRatio}:1</div>
                <div class="summary-label">Best Ratio</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">${summary.totalPairs}</div>
                <div class="summary-label">Pairs Tested</div>
            </div>
        </div>
    `;
    
    // Worst offenders
    const worstFailures = details.filter(d => d.compliance === 'FAIL').slice(0, 10);
    if (worstFailures.length > 0) {
        html += '<h4 style="color: #dc3545; margin: 15px 0 10px;">Failing Contrast Pairs (WCAG AA)</h4>';
        html += '<div class="contrast-details">';
        worstFailures.forEach(detail => {
            const fg = `rgb(${detail.fg.r}, ${detail.fg.g}, ${detail.fg.b})`;
            const bg = `rgb(${detail.bg.r}, ${detail.bg.g}, ${detail.bg.b})`;
            html += `
                <div class="contrast-row">
                    <div class="contrast-swatch" style="background: linear-gradient(90deg, ${fg} 50%, ${bg} 50%);"></div>
                    <div class="contrast-info">
                        <div class="contrast-ratio">${detail.ratio}:1</div>
                        <span class="contrast-compliance ${detail.compliance}">${detail.compliance}</span>
                    </div>
                    <div class="contrast-colors">
                        <div class="color-chip" style="background: ${fg};"></div>
                        <div class="color-chip" style="background: ${bg};"></div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
    }
    
    // Passing pairs (AA but not AAA)
    const aaOnly = details.filter(d => d.compliance === 'AA').slice(0, 5);
    if (aaOnly.length > 0) {
        html += '<h4 style="color: #ffc107; margin: 15px 0 10px;">Passing AA (Not AAA)</h4>';
        html += '<div class="contrast-details">';
        aaOnly.forEach(detail => {
            const fg = `rgb(${detail.fg.r}, ${detail.fg.g}, ${detail.fg.b})`;
            const bg = `rgb(${detail.bg.r}, ${detail.bg.g}, ${detail.bg.b})`;
            html += `
                <div class="contrast-row">
                    <div class="contrast-swatch" style="background: linear-gradient(90deg, ${fg} 50%, ${bg} 50%);"></div>
                    <div class="contrast-info">
                        <div class="contrast-ratio">${detail.ratio}:1</div>
                        <span class="contrast-compliance ${detail.compliance}">${detail.compliance}</span>
                    </div>
                    <div class="contrast-colors">
                        <div class="color-chip" style="background: ${fg};"></div>
                        <div class="color-chip" style="background: ${bg};"></div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
    }
    
    // Dominant colors palette
    if (dominantColors.length > 0) {
        html += '<h4 style="margin: 15px 0 10px;">Dominant Colors</h4>';
        html += '<div class="dominant-palette">';
        dominantColors.forEach((color, i) => {
            const rgb = `rgb(${color.r}, ${color.g}, ${color.b})`;
            html += `<div class="palette-color" style="background: ${rgb};" title="RGB: ${color.r}, ${color.g}, ${color.b}"></div>`;
        });
        html += '</div>';
    }
    
    // Actions
    html += `
        <div class="analysis-actions">
            <button class="btn btn-small" onclick="generateHeatmap()">Show Heatmap</button>
            <button class="btn btn-small secondary" onclick="exportAnalysis()">Export Report</button>
        </div>
    `;
    
    accessibilityContent.innerHTML = html;
    
    // Make functions globally accessible for onclick
    window.generateHeatmap = () => showHeatmap();
    window.exportAnalysis = () => exportAnalysisReport();
}

function showHeatmap() {
    if (!currentAnalysis || !currentImage) return;
    
    let imageData;
    if (gl && simulationProgram.program) {
        const pixels = new Uint8Array(canvas.width * canvas.height * 4);
        gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        const flipped = new Uint8ClampedArray(pixels.length);
        for (let y = 0; y < canvas.height; y++) {
            const srcRow = (canvas.height - 1 - y) * canvas.width * 4;
            const dstRow = y * canvas.width * 4;
            flipped.set(pixels.subarray(srcRow, srcRow + canvas.width * 4), dstRow);
        }
        imageData = new ImageData(flipped, canvas.width, canvas.height);
    } else {
        offscreenCtx.drawImage(currentImage, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        imageData = offscreenCtx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
    }
    
    const heatmap = generateContrastHeatmap(imageData, currentAnalysis, { opacity: 180 });
    ctx.putImageData(heatmap, 0, 0);
}

function exportAnalysisReport() {
    if (!currentAnalysis) return;
    
    const report = {
        timestamp: new Date().toISOString(),
        mode: currentMode,
        intensity: currentIntensity,
        summary: currentAnalysis.summary,
        details: currentAnalysis.details,
        dominantColors: currentAnalysis.dominantColors,
        wcagThresholds: WCAG_THRESHOLDS
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `contrast-analysis-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

function exportCurrentFrame() {
    if (!currentImage) return;
    
    // Render current frame to canvas if needed
    if (gl && simulationProgram.program) {
        renderWebGL();
    } else {
        renderCanvas2D();
    }
    
    // Export as PNG
    canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `bionic-vision-${currentMode}-${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(url);
    }, 'image/png');
}

// Initialize
function init() {
    console.log('Bionic Vision initializing...');
    
    const webglSupported = initWebGL();
    console.log('WebGL:', webglSupported ? 'enabled' : 'fallback to Canvas 2D');
    
    // Check for getUserMedia support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('getUserMedia not supported');
        toggleCameraBtn.disabled = true;
        toggleCameraBtn.title = 'Camera not supported in this browser';
    }
    
    resizeCanvas();
    updateModeUI();
    
    // Initial render (black screen)
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    console.log('Bionic Vision ready!');
}

init();