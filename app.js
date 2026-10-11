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
const splitLine = document.getElementById('splitLine');
const splitHandle = document.getElementById('splitHandle');

// Rendering Contexts
let gl = null;
let ctx = null; // Canvas 2D fallback context if WebGL is unavailable
const offscreenCtx = offscreenCanvas.getContext('2d');

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
let isDraggingSplit = false;
let isHeatmapActive = false;

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
    try {
        gl = canvas.getContext('webgl', { preserveDrawingBuffer: true, alpha: false }) ||
             canvas.getContext('experimental-webgl', { preserveDrawingBuffer: true, alpha: false });
    } catch (e) {
        console.warn('WebGL initialization exception:', e);
    }

    if (!gl) {
        console.warn('WebGL not supported, falling back to Canvas 2D');
        ctx = canvas.getContext('2d');
        return false;
    }

    // Create shader programs
    simulationProgram = new ShaderProgram(gl, VERTEX_SHADER_SOURCE, SIMULATION_FRAGMENT_SHADER_SOURCE);
    daltonizeProgram = new ShaderProgram(gl, VERTEX_SHADER_SOURCE, DALTONIZE_FRAGMENT_SHADER_SOURCE);
    pathologyProgram = new ShaderProgram(gl, VERTEX_SHADER_SOURCE, PATHOLOGY_FRAGMENT_SHADER_SOURCE);
    splitViewProgram = new ShaderProgram(gl, VERTEX_SHADER_SOURCE, SPLIT_VIEW_FRAGMENT_SHADER_SOURCE);

    if (!simulationProgram.program || !daltonizeProgram.program || !pathologyProgram.program || !splitViewProgram.program) {
        console.error('Failed to create shader programs, falling back to 2D');
        gl = null;
        ctx = canvas.getContext('2d');
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

// Clear canvas helper
function clearCanvas() {
    if (gl) {
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
    } else if (ctx) {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
}

// Resize canvas to match display size
function resizeCanvas() {
    const container = viewerContainer;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    let targetW = Math.max(320, Math.floor(rect.width * dpr));
    let targetH = Math.max(240, Math.floor((rect.height || 480) * dpr));

    if (currentImage) {
        const srcW = currentImage.videoWidth || currentImage.naturalWidth || currentImage.width;
        const srcH = currentImage.videoHeight || currentImage.naturalHeight || currentImage.height;
        if (srcW && srcH) {
            targetW = srcW;
            targetH = srcH;
        }
    }
    
    if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
        offscreenCanvas.width = targetW;
        offscreenCanvas.height = targetH;
    }
    
    if (gl) {
        gl.viewport(0, 0, canvas.width, canvas.height);
    }

    updateSplitLineUI();
}

// Update texture with video frame or image
function updateTexture(source) {
    if (!gl || !texture) return;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
}

// Setup quad geometry for shader program
function setupQuad(program) {
    const posLoc = program.getAttribLocation('a_position');
    const texLoc = program.getAttribLocation('a_texCoord');
    
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    
    gl.enableVertexAttribArray(texLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);
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
    if (isPathology) {
        program = pathologyProgram;
    } else if (useSplitView && !isNormal) {
        program = splitViewProgram;
    } else if (useDalton && simMatrix) {
        program = daltonizeProgram;
    } else {
        program = simulationProgram;
    }

    program.use();
    setupQuad(program);

    // Common uniforms
    gl.uniform1i(program.getUniformLocation('u_image'), 0);
    gl.uniform1f(program.getUniformLocation('u_intensity'), currentIntensity);

    const identityMatrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

    if (program === simulationProgram) {
        program.setUniformMatrix3fv('u_rgbToLms', RGB_TO_LMS);
        program.setUniformMatrix3fv('u_lmsToRgb', LMS_TO_RGB);
        gl.uniform1i(program.getUniformLocation('u_enabled'), simMatrix ? 1 : 0);
        if (simMatrix) {
            program.setUniformMatrix3fv('u_simMatrix', simMatrix);
        } else {
            program.setUniformMatrix3fv('u_simMatrix', identityMatrix);
        }
    } else if (program === daltonizeProgram) {
        program.setUniformMatrix3fv('u_rgbToLms', RGB_TO_LMS);
        program.setUniformMatrix3fv('u_lmsToRgb', LMS_TO_RGB);
        program.setUniformMatrix3fv('u_simMatrix', simMatrix || identityMatrix);
        const daltonMatrix = DALTONIZE_MATRICES[currentMode] || identityMatrix;
        gl.uniform1i(program.getUniformLocation('u_enabled'), 1);
        program.setUniformMatrix3fv('u_daltonizeMatrix', daltonMatrix);
    } else if (program === pathologyProgram) {
        const conditionMap = { 'cataracts': 1, 'glaucoma': 2, 'macular-degeneration': 3 };
        gl.uniform1i(program.getUniformLocation('u_condition'), conditionMap[currentMode] || 0);
        program.setUniform2f('u_resolution', canvas.width, canvas.height);
    } else if (program === splitViewProgram) {
        program.setUniformMatrix3fv('u_rgbToLms', RGB_TO_LMS);
        program.setUniformMatrix3fv('u_lmsToRgb', LMS_TO_RGB);
        
        gl.uniform1f(program.getUniformLocation('u_splitPos'), splitPosition);
        
        // Left side: uncorrected condition simulation
        gl.uniform1i(program.getUniformLocation('u_useLeftSim'), simMatrix ? 1 : 0);
        gl.uniform1i(program.getUniformLocation('u_useLeftDalton'), 0);
        program.setUniformMatrix3fv('u_simMatrixLeft', simMatrix || identityMatrix);
        program.setUniformMatrix3fv('u_daltonizeMatrixLeft', identityMatrix);

        // Right side: Daltonized simulation if Daltonization enabled, otherwise normal RGB
        const hasDalton = useDaltonization && DALTONIZE_MATRICES[currentMode];
        gl.uniform1i(program.getUniformLocation('u_useRightSim'), hasDalton ? 1 : 0);
        gl.uniform1i(program.getUniformLocation('u_useRightDalton'), hasDalton ? 1 : 0);
        program.setUniformMatrix3fv('u_simMatrixRight', simMatrix || identityMatrix);
        program.setUniformMatrix3fv('u_daltonizeMatrixRight', hasDalton ? DALTONIZE_MATRICES[currentMode] : identityMatrix);
    }

    gl.drawArrays(gl.TRIANGLES, 0, 6);
}

// Fallback Canvas 2D rendering (CPU)
function renderCanvas2D() {
    if (!ctx || !currentImage) return;

    const w = offscreenCanvas.width;
    const h = offscreenCanvas.height;

    // Draw source to offscreen
    offscreenCtx.drawImage(currentImage, 0, 0, w, h);
    const imageData = offscreenCtx.getImageData(0, 0, w, h);
    const data = imageData.data;

    const simMatrix = SIMULATION_MATRICES[currentMode];
    const isPathology = ['cataracts', 'glaucoma', 'macular-degeneration'].includes(currentMode);
    const isNormal = currentMode === 'normal';

    if (useSplitView && !isNormal && !isPathology) {
        const splitX = Math.floor(w * splitPosition);
        
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];

                let resultR = r, resultG = g, resultB = b;

                if (x < splitX) {
                    // Left side - uncorrected simulation
                    if (simMatrix) {
                        const sim = simulateColorblindness(r, g, b, currentMode, currentIntensity);
                        resultR = sim.r; resultG = sim.g; resultB = sim.b;
                    }
                } else {
                    // Right side - daltonization or normal
                    if (useDaltonization && DALTONIZE_MATRICES[currentMode]) {
                        const dalt = daltonize(r, g, b, currentMode, currentIntensity);
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
                } else if (useDaltonization && DALTONIZE_MATRICES[currentMode]) {
                    const dalt = daltonize(r, g, b, currentMode, currentIntensity);
                    resultR = dalt.r; resultG = dalt.g; resultB = dalt.b;
                } else if (simMatrix) {
                    const sim = simulateColorblindness(r, g, b, currentMode, currentIntensity);
                    resultR = sim.r; resultG = sim.g; resultB = sim.b;
                }
            }

            data[i] = resultR;
            data[i + 1] = resultG;
            data[i + 2] = resultB;
        }
    }

    ctx.putImageData(imageData, 0, 0);
}

function applyPathologyCPU(r, g, b, x, y, w, h, mode, intensity) {
    const centerX = w / 2;
    const centerY = h / 2;
    const dx = (x - centerX) / (w / 2);
    const dy = (y - centerY) / (h / 2);
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (mode === 'cataracts') {
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

// Render current frame with active pipeline
function renderCurrentFrame() {
    if (gl && simulationProgram && simulationProgram.program) {
        renderWebGL();
    } else {
        renderCanvas2D();
    }
}

// Animation loop
function animate() {
    if (!isCameraOn) return;

    if (videoElement && videoElement.readyState >= 2) {
        currentImage = videoElement;
    }
    
    if (currentImage && !isHeatmapActive) {
        renderCurrentFrame();
    }
    
    animationFrameId = requestAnimationFrame(animate);
}

// Helper to obtain mediaDevices getUserMedia across environments
function getCameraAccessFunction() {
    if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function') {
        return (constraints) => navigator.mediaDevices.getUserMedia(constraints);
    }
    const legacy = navigator.getUserMedia ||
                   navigator.webkitGetUserMedia ||
                   navigator.mozGetUserMedia ||
                   navigator.msGetUserMedia;
    if (legacy) {
        return (constraints) => new Promise((resolve, reject) => {
            legacy.call(navigator, constraints, resolve, reject);
        });
    }
    return null;
}

// Webcam handling
async function startCamera() {
    const getUserMediaFn = getCameraAccessFunction();
    if (!getUserMediaFn) {
        const isSecure = window.isSecureContext !== false;
        alert(
            'Unable to access camera: Web browser blocked camera API.\n\n' +
            (!isSecure ? 'Reason: Not in a Secure Context (browsers require HTTPS or http://localhost to use webcams).\n\n' : '') +
            'Solutions:\n' +
            '1. Access the app via http://localhost:8080/ or https://\n' +
            '2. If testing locally, run "npm start" and open http://localhost:8080/\n' +
            '3. Or click "Upload Image" to test all simulations and contrast analysis immediately!'
        );
        return;
    }

    try {
        loading.classList.add('active');
        console.log('Requesting camera access...');
        
        // Multi-tier constraint fallback: back camera -> ideal HD -> generic any video
        const constraintTiers = [
            { video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
            { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
            { video: true, audio: false }
        ];

        let stream = null;
        let lastError = null;

        for (const constraints of constraintTiers) {
            try {
                stream = await getUserMediaFn(constraints);
                if (stream) break;
            } catch (err) {
                lastError = err;
                console.warn('Camera tier failed:', constraints, err.name, err.message);
            }
        }

        if (!stream) {
            throw lastError || new Error('Could not access any camera devices.');
        }

        videoStream = stream;

        // Create and configure video element
        videoElement = document.createElement('video');
        videoElement.autoplay = true;
        videoElement.muted = true;
        videoElement.playsInline = true;
        videoElement.setAttribute('autoplay', '');
        videoElement.setAttribute('muted', '');
        videoElement.setAttribute('playsinline', '');
        videoElement.setAttribute('webkit-playsinline', '');

        // Attaching to DOM ensures WebKit / Blink hardware video decoder is active
        videoElement.style.position = 'fixed';
        videoElement.style.top = '-9999px';
        videoElement.style.left = '-9999px';
        videoElement.style.width = '1px';
        videoElement.style.height = '1px';
        videoElement.style.opacity = '0';
        videoElement.style.pointerEvents = 'none';
        document.body.appendChild(videoElement);

        videoElement.srcObject = videoStream;

        // Ensure video is playing and ready with frame dimensions
        await new Promise((resolve, reject) => {
            let done = false;
            const timeoutId = setTimeout(() => {
                if (!done) {
                    done = true;
                    cleanup();
                    if (videoElement.videoWidth > 0) {
                        resolve();
                    } else {
                        reject(new Error('Timed out waiting for camera frames.'));
                    }
                }
            }, 8000);

            const cleanup = () => {
                clearTimeout(timeoutId);
                videoElement.removeEventListener('loadedmetadata', onMetadata);
                videoElement.removeEventListener('canplay', onReady);
                videoElement.removeEventListener('error', onError);
            };

            const onMetadata = async () => {
                try {
                    await videoElement.play();
                } catch (playErr) {
                    console.warn('Play attempt after metadata:', playErr);
                }
                if (videoElement.videoWidth > 0 && !done) {
                    done = true;
                    cleanup();
                    resolve();
                }
            };

            const onReady = async () => {
                try {
                    await videoElement.play();
                } catch (playErr) {
                    console.warn('Play attempt after canplay:', playErr);
                }
                if (!done) {
                    done = true;
                    cleanup();
                    resolve();
                }
            };

            const onError = (e) => {
                if (!done) {
                    done = true;
                    cleanup();
                    reject(e || new Error('Video stream element error.'));
                }
            };

            videoElement.addEventListener('loadedmetadata', onMetadata);
            videoElement.addEventListener('canplay', onReady);
            videoElement.addEventListener('error', onError);

            // Attempt immediate play
            videoElement.play().catch(() => {});

            if (videoElement.readyState >= 2 && videoElement.videoWidth > 0) {
                done = true;
                cleanup();
                resolve();
            }
        });

        console.log('Camera started successfully:', videoElement.videoWidth, 'x', videoElement.videoHeight);
        
        currentImage = videoElement;
        isCameraOn = true;
        toggleCameraBtn.textContent = 'Stop Webcam';
        toggleCameraBtn.classList.add('active');
        uploadBtn.disabled = true;
        isHeatmapActive = false;
        
        resizeCanvas();
        animate();
        
        loading.classList.remove('active');
    } catch (err) {
        console.error('Camera error:', err);
        loading.classList.remove('active');
        stopCamera();

        let friendlyMsg = err.message;
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            friendlyMsg = 'Camera permission was denied. Please click the lock/camera icon in your address bar and allow camera access.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            friendlyMsg = 'No camera found on this device. Please connect a webcam or use "Upload Image".';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            friendlyMsg = 'Camera is in use by another app (Zoom, Teams, or another tab). Please close other apps and try again.';
        } else if (err.name === 'OverconstrainedError') {
            friendlyMsg = 'Camera resolution constraints not supported by your hardware.';
        }

        alert('Unable to start webcam: ' + friendlyMsg + '\n\nYou can also click "Upload Image" to test any photo!');
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => {
            try { track.stop(); } catch (e) {}
        });
        videoStream = null;
    }
    if (videoElement) {
        try { videoElement.pause(); } catch (e) {}
        videoElement.srcObject = null;
        try { videoElement.remove(); } catch (e) {}
        videoElement = null;
    }
    currentImage = null;
    toggleCameraBtn.textContent = 'Start Webcam';
    toggleCameraBtn.classList.remove('active');
    isCameraOn = false;
    uploadBtn.disabled = false;
    isHeatmapActive = false;
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    clearCanvas();
}

// Image upload handling
function handleFileUpload(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
    }

    loading.classList.add('active');
    
    const img = new Image();
    img.onload = () => {
        stopCamera();
        currentImage = img;
        isHeatmapActive = false;
        resizeCanvas();
        renderCurrentFrame();
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
    
    // Disable daltonization & split view for pathologies or normal
    if (isPathology || isNormal) {
        daltonizationCheckbox.checked = false;
        useDaltonization = false;
        splitViewCheckbox.checked = false;
        useSplitView = false;
    }

    // Show/hide split slider
    splitSliderGroup.style.display = (useSplitView && !isNormal && !isPathology) ? 'block' : 'none';

    // Update condition info
    const info = CONDITION_INFO[currentMode];
    if (info) {
        conditionInfo.innerHTML = `
            <h4>${info.name}</h4>
            <p>${info.description}</p>
            <p><strong>Prevalence:</strong> ${info.prevalence}</p>
        `;
    }

    updateSplitLineUI();
}

// Update DOM Split Line UI
function updateSplitLineUI() {
    if (!splitLine) return;
    const isPathology = ['cataracts', 'glaucoma', 'macular-degeneration'].includes(currentMode);
    if (useSplitView && currentMode !== 'normal' && !isPathology) {
        splitLine.style.display = 'block';
        splitLine.style.left = `${splitPosition * 100}%`;
    } else {
        splitLine.style.display = 'none';
    }
}

// Update split position from client pointer X
function updateSplitFromPointer(clientX) {
    const rect = viewerContainer.getBoundingClientRect();
    if (rect.width > 0) {
        const x = clientX - rect.left;
        splitPosition = Math.max(0.05, Math.min(0.95, x / rect.width));
        splitSlider.value = Math.round(splitPosition * 100);
        updateSplitLineUI();
        if (currentImage && !isCameraOn) {
            renderCurrentFrame();
        }
    }
}

// Event Listeners
modeSelect.addEventListener('change', (e) => {
    currentMode = e.target.value;
    updateModeUI();
    if (currentImage && !isCameraOn) {
        renderCurrentFrame();
    }
});

intensitySlider.addEventListener('input', (e) => {
    currentIntensity = e.target.value / 100;
    intensityValue.textContent = e.target.value + '%';
    if (currentImage && !isCameraOn) {
        renderCurrentFrame();
    }
});

daltonizationCheckbox.addEventListener('change', (e) => {
    useDaltonization = e.target.checked;
    updateModeUI();
    if (currentImage && !isCameraOn) {
        renderCurrentFrame();
    }
});

splitViewCheckbox.addEventListener('change', (e) => {
    useSplitView = e.target.checked;
    updateModeUI();
    if (currentImage && !isCameraOn) {
        renderCurrentFrame();
    }
});

splitSlider.addEventListener('input', (e) => {
    splitPosition = e.target.value / 100;
    updateSplitLineUI();
    if (currentImage && useSplitView && !isCameraOn) {
        renderCurrentFrame();
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
    if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
    }
});

analyzeBtn.addEventListener('click', analyzeCurrentFrame);
exportBtn.addEventListener('click', exportCurrentFrame);

// Split line dragging - mouse and touch support
function startDrag(e) {
    if (!useSplitView || currentMode === 'normal') return;
    isDraggingSplit = true;
    document.body.style.cursor = 'ew-resize';
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    updateSplitFromPointer(clientX);
    if (e.cancelable) e.preventDefault();
}

function handleDrag(e) {
    if (!isDraggingSplit) return;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    updateSplitFromPointer(clientX);
    if (e.cancelable) e.preventDefault();
}

function endDrag() {
    if (isDraggingSplit) {
        isDraggingSplit = false;
        document.body.style.cursor = 'default';
    }
}

if (splitHandle) {
    splitHandle.addEventListener('mousedown', startDrag);
    splitHandle.addEventListener('touchstart', startDrag, { passive: false });
}

window.addEventListener('mousemove', handleDrag);
window.addEventListener('mouseup', endDrag);
window.addEventListener('touchmove', handleDrag, { passive: false });
window.addEventListener('touchend', endDrag);

// Window resize
window.addEventListener('resize', () => {
    resizeCanvas();
    if (currentImage && !isCameraOn) {
        renderCurrentFrame();
    }
});

// Contrast Analysis
async function analyzeCurrentFrame() {
    if (isAnalyzing) return;
    if (!currentImage) {
        alert('Please start the webcam or upload an image first.');
        return;
    }
    
    isAnalyzing = true;
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    accessibilityPanel.style.display = 'block';
    analysisStatus.textContent = 'Scanning...';
    analysisStatus.className = 'status-badge pending';
    accessibilityContent.innerHTML = '<p>Analyzing contrast ratios...</p>';
    
    await new Promise(r => setTimeout(r, 50));
    
    try {
        let imageData;
        if (gl && simulationProgram && simulationProgram.program) {
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
        
        currentAnalysis = analyzeContrast(imageData, {
            sampleStep: 4,
            edgeThreshold: 25,
            maxEdgeSamples: 300
        });
        
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
    
    let overallStatus = 'fail';
    let statusText = 'FAIL';
    if (summary.totalPairs > 0 && summary.aaaPass > summary.totalPairs * 0.5) {
        overallStatus = 'pass-aaa';
        statusText = 'AAA PASS';
    } else if (summary.totalPairs > 0 && summary.aaPass > summary.totalPairs * 0.5) {
        overallStatus = 'pass-aa';
        statusText = 'AA PASS';
    }
    
    analysisStatus.textContent = statusText;
    analysisStatus.className = `status-badge ${overallStatus}`;
    
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
    
    if (dominantColors.length > 0) {
        html += '<h4 style="margin: 15px 0 10px;">Dominant Colors</h4>';
        html += '<div class="dominant-palette">';
        dominantColors.forEach((color) => {
            const rgb = `rgb(${color.r}, ${color.g}, ${color.b})`;
            html += `<div class="palette-color" style="background: ${rgb};" title="RGB: ${color.r}, ${color.g}, ${color.b}"></div>`;
        });
        html += '</div>';
    }
    
    html += `
        <div class="analysis-actions">
            <button id="heatmapToggleBtn" class="btn btn-small">${isHeatmapActive ? 'Hide Heatmap' : 'Show Heatmap'}</button>
            <button id="exportReportBtn" class="btn btn-small secondary">Export Report</button>
        </div>
    `;
    
    accessibilityContent.innerHTML = html;

    const heatmapToggleBtn = document.getElementById('heatmapToggleBtn');
    if (heatmapToggleBtn) {
        heatmapToggleBtn.addEventListener('click', toggleHeatmap);
    }
    const exportReportBtn = document.getElementById('exportReportBtn');
    if (exportReportBtn) {
        exportReportBtn.addEventListener('click', exportAnalysisReport);
    }
}

function toggleHeatmap() {
    if (!currentAnalysis || !currentImage) {
        alert('Please analyze contrast first.');
        return;
    }
    
    isHeatmapActive = !isHeatmapActive;
    const heatmapToggleBtn = document.getElementById('heatmapToggleBtn');
    if (heatmapToggleBtn) {
        heatmapToggleBtn.textContent = isHeatmapActive ? 'Hide Heatmap' : 'Show Heatmap';
    }
    
    if (isHeatmapActive) {
        let imageData;
        if (gl && simulationProgram && simulationProgram.program) {
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
        
        if (gl && simulationProgram && simulationProgram.program) {
            updateTexture(heatmap);
            gl.clearColor(0, 0, 0, 1);
            gl.clear(gl.COLOR_BUFFER_BIT);
            simulationProgram.use();
            setupQuad(simulationProgram);
            gl.uniform1i(simulationProgram.getUniformLocation('u_image'), 0);
            gl.uniform1i(simulationProgram.getUniformLocation('u_enabled'), 0);
            gl.drawArrays(gl.TRIANGLES, 0, 6);
        } else if (ctx) {
            ctx.putImageData(heatmap, 0, 0);
        }
    } else {
        renderCurrentFrame();
    }
}

function exportAnalysisReport() {
    if (!currentAnalysis) {
        alert('Please analyze contrast first.');
        return;
    }
    
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
    if (!currentImage) {
        alert('Please start the webcam or upload an image first.');
        return;
    }
    
    renderCurrentFrame();
    
    canvas.toBlob((blob) => {
        if (!blob) {
            alert('Failed to generate image file.');
            return;
        }
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
    console.log('WebGL:', webglSupported ? 'enabled (GPU accelerated)' : 'fallback to Canvas 2D (CPU)');
    
    // Check for getUserMedia support
    if (!getCameraAccessFunction()) {
        console.warn('getUserMedia not supported in this context (HTTPS or localhost required for camera)');
        toggleCameraBtn.title = 'Camera requires a secure connection (HTTPS) or localhost';
    }
    
    resizeCanvas();
    updateModeUI();
    clearCanvas();
    
    // Check URL parameters for PWA shortcuts
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('camera') === '1') {
        startCamera();
    } else if (urlParams.get('upload') === '1') {
        fileInput.click();
    }

    // Window globals for inline button fallbacks
    window.generateHeatmap = toggleHeatmap;
    window.exportAnalysis = exportAnalysisReport;
    
    console.log('Bionic Vision ready!');
}

init();