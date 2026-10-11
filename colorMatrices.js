// RGB to LMS conversion matrix (Stockman & Sharpe 2° cone fundamentals, normalized)
const RGB_TO_LMS = [
    [0.31399022, 0.63951294, 0.04649755],
    [0.15537241, 0.75789446, 0.08670142],
    [0.01775239, 0.10944209, 0.87256922]
];

// LMS to RGB conversion matrix (inverse of above)
const LMS_TO_RGB = [
    [5.47221206, -4.64196010, 0.16963708],
    [-1.12524190, 2.29317094, -0.16789520],
    [0.02980165, -0.19318073, 1.16364789]
];

// Daltonization matrices for red-green colorblindness
const PROTANOPIA_SIM = [
    [0.0, 2.02352086, -2.52541059],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
];

const DEUTERANOPIA_SIM = [
    [1.0, 0.0, 0.0],
    [0.49421479, 0.0, 1.24880799],
    [0.0, 0.0, 1.0]
];

const TRITANOPIA_SIM = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-0.01224588, 0.07203564, 0.0]
];

// Anomalous trichromacy (weakened cones)
const PROTANOMALY_SIM = [
    [0.4655, 0.4670, 0.0675],
    [0.1843, 0.8097, 0.0060],
    [0.0047, 0.0437, 0.9516]
];

const DEUTERANOMALY_SIM = [
    [0.4615, 0.5070, 0.0315],
    [0.1993, 0.7943, 0.0064],
    [0.0053, 0.0387, 0.9560]
];

const TRITANOMALY_SIM = [
    [0.9205, 0.0263, 0.0532],
    [0.0200, 0.9574, 0.0226],
    [0.0367, 0.1283, 0.8350]
];

// Achromatopsia (no color)
const ACHROMATOPSIA_SIM = [
    [0.299, 0.587, 0.114],
    [0.299, 0.587, 0.114],
    [0.299, 0.587, 0.114]
];

const ACHROMATOMALY_SIM = [
    [0.618, 0.320, 0.062],
    [0.163, 0.775, 0.062],
    [0.163, 0.320, 0.517]
];

// Enhanced Daltonization using Machado et al. algorithm
// Computes the error (confusion) in LMS space and projects it to the visible dimension
// This preserves luminance while maximizing chromatic discrimination

// Error projection matrices for each deficiency type (CPU enhanced version)
// These map the lost LMS dimension to the remaining two dimensions
const DALTONIZE_ERROR_PROJECTION = {
    // For Protan: L-cone missing, project L-error onto M and S
    'protan': [
        [0, 0, 0],
        [0.7, 0, 0],  // L-error -> M
        [0.7, 0, 0]   // L-error -> S
    ],
    // For Deutan: M-cone missing, project M-error onto L and S
    'deutan': [
        [0, 0.7, 0],  // M-error -> L
        [0, 0, 0],
        [0, 0.7, 0]   // M-error -> S
    ],
    // For Tritan: S-cone missing, project S-error onto L and M
    'tritan': [
        [0, 0, 0.7],  // S-error -> L
        [0, 0, 0.7],  // S-error -> M
        [0, 0, 0]
    ]
};

// Simple matrix-based daltonization for WebGL shaders (retains base channel with 1.0 on diagonal)
const DALTONIZE_PROTAN = [
    [1.0, 0.0, 0.0],
    [0.5, 1.0, 0.0],
    [0.5, 0.0, 1.0]
];

const DALTONIZE_DEUTERAN = [
    [1.0, 0.5, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.5, 1.0]
];

const DALTONIZE_TRITAN = [
    [1.0, 0.0, 0.5],
    [0.0, 1.0, 0.5],
    [0.0, 0.0, 1.0]
];

function getDeficiencyType(mode) {
    if (mode.startsWith('protan')) return 'protan';
    if (mode.startsWith('deutan')) return 'deutan';
    if (mode.startsWith('tritan')) return 'tritan';
    return null;
}

// Enhanced Daltonization (CPU): Compute simulation error in LMS space, project to visible dimensions
function daltonizeEnhanced(r, g, b, mode, intensity = 1.0) {
    const defType = getDeficiencyType(mode);
    if (!defType) return { r, g, b };

    const simMatrix = SIMULATION_MATRICES[mode];
    if (!simMatrix) return { r, g, b };

    const rgb = [r / 255, g / 255, b / 255];
    
    // Convert to LMS
    const lms = multiplyMatrixVector(RGB_TO_LMS, rgb);
    
    // Apply simulation
    const simLms = multiplyMatrixVector(simMatrix, lms);
    
    // Compute error (what was lost)
    const errorLms = [
        lms[0] - simLms[0],
        lms[1] - simLms[1],
        lms[2] - simLms[2]
    ];
    
    // Project error to visible dimensions
    const projection = DALTONIZE_ERROR_PROJECTION[defType];
    const correctionLms = multiplyMatrixVector(projection, errorLms);
    
    // Add correction to simulated LMS
    const correctedLms = [
        simLms[0] + correctionLms[0],
        simLms[1] + correctionLms[1],
        simLms[2] + correctionLms[2]
    ];
    
    // Convert back to RGB
    const correctedRgb = multiplyMatrixVector(LMS_TO_RGB, correctedLms);
    
    // Blend with original based on intensity
    return {
        r: clamp((correctedRgb[0] * intensity + rgb[0] * (1 - intensity)) * 255, 0, 255),
        g: clamp((correctedRgb[1] * intensity + rgb[1] * (1 - intensity)) * 255, 0, 255),
        b: clamp((correctedRgb[2] * intensity + rgb[2] * (1 - intensity)) * 255, 0, 255)
    };
}

// Main daltonize function - uses enhanced CPU version
function daltonize(r, g, b, mode, intensity = 1.0) {
    // Use enhanced version for better results
    return daltonizeEnhanced(r, g, b, mode, intensity);
}

// Map simulation modes to their matrices
const SIMULATION_MATRICES = {
    'normal': null,
    'protanopia': PROTANOPIA_SIM,
    'protanomaly': PROTANOMALY_SIM,
    'deuteranopia': DEUTERANOPIA_SIM,
    'deuteranomaly': DEUTERANOMALY_SIM,
    'tritanopia': TRITANOPIA_SIM,
    'tritanomaly': TRITANOMALY_SIM,
    'achromatopsia': ACHROMATOPSIA_SIM,
    'achromatomaly': ACHROMATOMALY_SIM,
};

// Simple matrices for WebGL shader compatibility
const DALTONIZE_MATRICES = {
    'protanopia': DALTONIZE_PROTAN,
    'protanomaly': DALTONIZE_PROTAN,
    'deuteranopia': DALTONIZE_DEUTERAN,
    'deuteranomaly': DALTONIZE_DEUTERAN,
    'tritanopia': DALTONIZE_TRITAN,
    'tritanomaly': DALTONIZE_TRITAN,
};

function multiplyMatrixVector(matrix, vector) {
    return [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2]
    ];
}

function multiplyMatrices(a, b) {
    const result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            for (let k = 0; k < 3; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function simulateColorblindness(r, g, b, mode, intensity = 1.0) {
    const simMatrix = SIMULATION_MATRICES[mode];
    if (!simMatrix) return { r, g, b };

    // Convert RGB to LMS
    const rgb = [r / 255, g / 255, b / 255];
    const lms = multiplyMatrixVector(RGB_TO_LMS, rgb);

    // Apply simulation matrix in LMS space
    const simLms = multiplyMatrixVector(simMatrix, lms);

    // Convert back to RGB
    const simRgb = multiplyMatrixVector(LMS_TO_RGB, simLms);

    // Blend with original based on intensity
    return {
        r: clamp((simRgb[0] * intensity + rgb[0] * (1 - intensity)) * 255, 0, 255),
        g: clamp((simRgb[1] * intensity + rgb[1] * (1 - intensity)) * 255, 0, 255),
        b: clamp((simRgb[2] * intensity + rgb[2] * (1 - intensity)) * 255, 0, 255)
    };
}

function createCombinedMatrix(mode) {
    const simMatrix = SIMULATION_MATRICES[mode];
    if (!simMatrix) return LMS_TO_RGB; // Identity-like for normal

    return multiplyMatrices(LMS_TO_RGB, multiplyMatrices(simMatrix, RGB_TO_LMS));
}

export {
    RGB_TO_LMS,
    LMS_TO_RGB,
    SIMULATION_MATRICES,
    DALTONIZE_MATRICES,
    simulateColorblindness,
    daltonize,
    daltonizeEnhanced,
    createCombinedMatrix,
    multiplyMatrixVector,
    multiplyMatrices
};