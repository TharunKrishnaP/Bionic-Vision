// WebGL Shaders for Real-time Color Vision Simulation
// Uses fragment shaders to run matrix calculations on GPU at 60 FPS

// Vertex Shader - passes through position and texture coordinates
const VERTEX_SHADER_SOURCE = `
    attribute vec2 a_position;
    attribute vec2 a_texCoord;
    varying vec2 v_texCoord;
    
    void main() {
        v_texCoord = a_texCoord;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

// Fragment Shader for Colorblindness Simulation
const SIMULATION_FRAGMENT_SHADER_SOURCE = `
    precision highp float;
    
    uniform sampler2D u_image;
    uniform mat3 u_rgbToLms;
    uniform mat3 u_simMatrix;
    uniform mat3 u_lmsToRgb;
    uniform float u_intensity;
    uniform bool u_enabled;
    varying vec2 v_texCoord;
    
    void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        if (!u_enabled) {
            gl_FragColor = color;
            return;
        }
        
        vec3 rgb = color.rgb;
        
        // Convert RGB to LMS
        vec3 lms = u_rgbToLms * rgb;
        
        // Apply simulation matrix
        vec3 simLms = u_simMatrix * lms;
        
        // Convert back to RGB
        vec3 simRgb = u_lmsToRgb * simLms;
        
        // Blend with original based on intensity
        vec3 finalRgb = mix(rgb, simRgb, u_intensity);
        
        gl_FragColor = vec4(finalRgb, color.a);
    }
`;

// Fragment Shader for Daltonization Correction
const DALTONIZE_FRAGMENT_SHADER_SOURCE = `
    precision highp float;
    
    uniform sampler2D u_image;
    uniform mat3 u_rgbToLms;
    uniform mat3 u_simMatrix;
    uniform mat3 u_lmsToRgb;
    uniform mat3 u_daltonizeMatrix;
    uniform float u_intensity;
    uniform bool u_enabled;
    varying vec2 v_texCoord;
    
    void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        if (!u_enabled) {
            gl_FragColor = color;
            return;
        }
        
        vec3 rgb = color.rgb;
        
        // Apply daltonization matrix directly in RGB space
        vec3 corrected = u_daltonizeMatrix * rgb;
        
        // Simulate how the corrected colors are perceived under the color vision deficiency
        vec3 simLms = u_simMatrix * (u_rgbToLms * corrected);
        vec3 simRgb = u_lmsToRgb * simLms;
        
        // Blend with original based on intensity
        vec3 finalRgb = mix(rgb, simRgb, u_intensity);
        
        gl_FragColor = vec4(finalRgb, color.a);
    }
`;

// Fragment Shader for Pathology Simulations (Cataracts, Glaucoma, Macular Degeneration)
const PATHOLOGY_FRAGMENT_SHADER_SOURCE = `
    precision highp float;
    
    uniform sampler2D u_image;
    uniform int u_condition; // 0=none, 1=cataracts, 2=glaucoma, 3=macular
    uniform float u_intensity;
    uniform vec2 u_resolution;
    varying vec2 v_texCoord;
    
    // Gaussian blur for cataracts (constant loop bounds for WebGL 1.0 / GLSL ES 1.00 compliance)
    vec3 gaussianBlur(sampler2D tex, vec2 uv, float sigma) {
        vec3 color = vec3(0.0);
        float totalWeight = 0.0;
        vec2 texelSize = 1.0 / max(u_resolution, vec2(1.0));
        float s = max(0.1, sigma);
        float twoSigmaSq = 2.0 * s * s;
        
        for (float x = -5.0; x <= 5.0; x += 1.0) {
            for (float y = -5.0; y <= 5.0; y += 1.0) {
                float distSq = x * x + y * y;
                float weight = exp(-distSq / twoSigmaSq);
                vec2 offset = vec2(x, y) * texelSize * (s * 0.4 + 0.6);
                color += texture2D(tex, uv + offset).rgb * weight;
                totalWeight += weight;
            }
        }
        return totalWeight > 0.0 ? color / totalWeight : texture2D(tex, uv).rgb;
    }
    
    void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        vec3 result = color.rgb;
        vec2 center = v_texCoord - 0.5;
        
        if (u_condition == 1) { // Cataracts
            float sigma = u_intensity * 4.0;
            vec3 blurred = gaussianBlur(u_image, v_texCoord, sigma);
            // Also reduce contrast
            vec3 gray = vec3(dot(blurred, vec3(0.299, 0.587, 0.114)));
            result = mix(blurred, gray, u_intensity * 0.3);
        } 
        else if (u_condition == 2) { // Glaucoma - Tunnel Vision
            float dist = length(center * 2.0);
            float innerRadius = 0.3 + (1.0 - u_intensity) * 0.4;
            float outerRadius = 0.7 + (1.0 - u_intensity) * 0.3;
            float vignette = smoothstep(innerRadius, outerRadius, dist);
            result = mix(color.rgb, vec3(0.0), vignette * u_intensity);
        } 
        else if (u_condition == 3) { // Macular Degeneration - Central Blur
            float dist = length(center * 2.0);
            float blurRadius = u_intensity * 0.15;
            if (dist < blurRadius) {
                float sigma = (blurRadius - dist) * 10.0;
                result = gaussianBlur(u_image, v_texCoord, sigma);
            }
            // Add slight distortion in center
            float distortion = (1.0 - smoothstep(0.0, blurRadius, dist)) * u_intensity * 0.1;
            result = mix(color.rgb, result, 1.0 - distortion);
        }
        
        gl_FragColor = vec4(result, color.a);
    }
`;

// Split View Shader - renders two different simulations side by side
const SPLIT_VIEW_FRAGMENT_SHADER_SOURCE = `
    precision highp float;
    
    uniform sampler2D u_image;
    uniform mat3 u_rgbToLms;
    uniform mat3 u_simMatrixLeft;
    uniform mat3 u_simMatrixRight;
    uniform mat3 u_lmsToRgb;
    uniform mat3 u_daltonizeMatrixLeft;
    uniform mat3 u_daltonizeMatrixRight;
    uniform float u_intensity;
    uniform float u_splitPos; // 0.0 to 1.0
    uniform bool u_useLeftSim;
    uniform bool u_useRightSim;
    uniform bool u_useLeftDalton;
    uniform bool u_useRightDalton;
    varying vec2 v_texCoord;
    
    vec3 applySimulation(vec3 rgb, mat3 simMatrix) {
        vec3 lms = u_rgbToLms * rgb;
        vec3 simLms = simMatrix * lms;
        return u_lmsToRgb * simLms;
    }
    
    vec3 applyDaltonizeAndSim(vec3 rgb, mat3 daltonMatrix, mat3 simMatrix) {
        vec3 daltonRgb = daltonMatrix * rgb;
        vec3 lms = u_rgbToLms * daltonRgb;
        vec3 simLms = simMatrix * lms;
        return u_lmsToRgb * simLms;
    }
    
    void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        vec3 rgb = color.rgb;
        vec3 leftResult = rgb;
        vec3 rightResult = rgb;
        
        // Left side processing
        if (u_useLeftSim) {
            if (u_useLeftDalton) {
                leftResult = applyDaltonizeAndSim(rgb, u_daltonizeMatrixLeft, u_simMatrixLeft);
            } else {
                leftResult = applySimulation(rgb, u_simMatrixLeft);
            }
            leftResult = mix(rgb, leftResult, u_intensity);
        }
        
        // Right side processing
        if (u_useRightSim) {
            if (u_useRightDalton) {
                rightResult = applyDaltonizeAndSim(rgb, u_daltonizeMatrixRight, u_simMatrixRight);
            } else {
                rightResult = applySimulation(rgb, u_simMatrixRight);
            }
            rightResult = mix(rgb, rightResult, u_intensity);
        }
        
        // Split view
        vec3 finalRgb = (v_texCoord.x < u_splitPos) ? leftResult : rightResult;
        
        gl_FragColor = vec4(finalRgb, color.a);
    }
`;

// Shader Program Manager
class ShaderProgram {
    constructor(gl, vertexSource, fragmentSource) {
        this.gl = gl;
        this.program = this.createProgram(vertexSource, fragmentSource);
        this.attributes = {};
        this.uniforms = {};
    }
    
    createProgram(vertexSource, fragmentSource) {
        const gl = this.gl;
        const vertexShader = this.createShader(gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, fragmentSource);
        
        if (!vertexShader || !fragmentShader) return null;
        
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(program));
            return null;
        }
        
        return program;
    }
    
    createShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        
        return shader;
    }
    
    use() {
        this.gl.useProgram(this.program);
    }
    
    getAttribLocation(name) {
        if (this.attributes[name] === undefined) {
            this.attributes[name] = this.gl.getAttribLocation(this.program, name);
        }
        return this.attributes[name];
    }
    
    getUniformLocation(name) {
        if (typeof name !== 'string') return name; // In case a WebGLUniformLocation is passed directly
        if (this.uniforms[name] === undefined) {
            this.uniforms[name] = this.gl.getUniformLocation(this.program, name);
        }
        return this.uniforms[name];
    }
    
    setUniformMatrix3fv(name, matrix) {
        const loc = this.getUniformLocation(name);
        if (!loc || !matrix) return;
        // Transpose row-major (JavaScript) to column-major (WebGL GLSL)
        const colMajor = new Float32Array([
            matrix[0][0], matrix[1][0], matrix[2][0],
            matrix[0][1], matrix[1][1], matrix[2][1],
            matrix[0][2], matrix[1][2], matrix[2][2]
        ]);
        this.gl.uniformMatrix3fv(loc, false, colMajor);
    }
    
    setUniform1f(name, value) {
        const loc = this.getUniformLocation(name);
        if (loc !== null && loc !== undefined) {
            this.gl.uniform1f(loc, value);
        }
    }
    
    setUniform1i(name, value) {
        const loc = this.getUniformLocation(name);
        if (loc !== null && loc !== undefined) {
            this.gl.uniform1i(loc, value);
        }
    }
    
    setUniform2f(name, x, y) {
        const loc = this.getUniformLocation(name);
        if (loc !== null && loc !== undefined) {
            this.gl.uniform2f(loc, x, y);
        }
    }
    
    setUniform1iv(name, array) {
        const loc = this.getUniformLocation(name);
        if (loc !== null && loc !== undefined) {
            this.gl.uniform1iv(loc, array);
        }
    }
}

export {
    VERTEX_SHADER_SOURCE,
    SIMULATION_FRAGMENT_SHADER_SOURCE,
    DALTONIZE_FRAGMENT_SHADER_SOURCE,
    PATHOLOGY_FRAGMENT_SHADER_SOURCE,
    SPLIT_VIEW_FRAGMENT_SHADER_SOURCE,
    ShaderProgram
};