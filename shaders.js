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
        if (!u_enabled) {
            gl_FragColor = texture2D(u_image, v_texCoord);
            return;
        }
        
        vec4 color = texture2D(u_image, v_texCoord);
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
    uniform mat3 u_daltonizeMatrix;
    uniform float u_intensity;
    uniform bool u_enabled;
    varying vec2 v_texCoord;
    
    void main() {
        if (!u_enabled) {
            gl_FragColor = texture2D(u_image, v_texCoord);
            return;
        }
        
        vec4 color = texture2D(u_image, v_texCoord);
        vec3 rgb = color.rgb;
        
        // Apply daltonization matrix directly in RGB space
        vec3 corrected = u_daltonizeMatrix * rgb;
        
        // Blend with original based on intensity
        vec3 finalRgb = mix(rgb, corrected, u_intensity);
        
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
    
    // Gaussian blur for cataracts
    vec3 gaussianBlur(sampler2D tex, vec2 uv, float sigma) {
        vec3 color = vec3(0.0);
        float totalWeight = 0.0;
        float radius = sigma * 3.0;
        vec2 texelSize = 1.0 / u_resolution;
        
        for (float x = -radius; x <= radius; x += 1.0) {
            for (float y = -radius; y <= radius; y += 1.0) {
                vec2 offset = vec2(x, y) * texelSize;
                float weight = exp(-(x*x + y*y) / (2.0 * sigma * sigma));
                color += texture2D(tex, uv + offset).rgb * weight;
                totalWeight += weight;
            }
        }
        return color / totalWeight;
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
    
    vec3 applyDaltonize(vec3 rgb, mat3 daltonMatrix) {
        return daltonMatrix * rgb;
    }
    
    void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        vec3 rgb = color.rgb;
        vec3 leftResult = rgb;
        vec3 rightResult = rgb;
        
        // Left side processing
        if (u_useLeftSim) {
            leftResult = applySimulation(rgb, u_simMatrixLeft);
            if (u_useLeftDalton) {
                leftResult = applyDaltonize(leftResult, u_daltonizeMatrixLeft);
            }
            leftResult = mix(rgb, leftResult, u_intensity);
        }
        
        // Right side processing
        if (u_useRightSim) {
            rightResult = applySimulation(rgb, u_simMatrixRight);
            if (u_useRightDalton) {
                rightResult = applyDaltonize(rightResult, u_daltonizeMatrixRight);
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
        if (this.uniforms[name] === undefined) {
            this.uniforms[name] = this.gl.getUniformLocation(this.program, name);
        }
        return this.uniforms[name];
    }
    
    setUniformMatrix3fv(name, matrix) {
        this.gl.uniformMatrix3fv(this.getUniformLocation(name), false, new Float32Array(matrix.flat()));
    }
    
    setUniform1f(name, value) {
        this.gl.uniform1f(this.getUniformLocation(name), value);
    }
    
    setUniform1i(name, value) {
        this.gl.uniform1i(this.getUniformLocation(name), value);
    }
    
    setUniform2f(name, x, y) {
        this.gl.uniform2f(this.getUniformLocation(name), x, y);
    }
    
    setUniform1iv(name, array) {
        this.gl.uniform1iv(this.getUniformLocation(name), array);
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