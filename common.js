function vec2(x, y) {
    return [x, y];
}

function vec3(x, y, z) {
    return [x, y, z];
}

function vec4(x, y, z, w) {
    return [x, y, z, w];
}

function mat2(m00, m01, m10, m11) {
    return [
        [m00, m01],
        [m10, m11]
    ];
}

function mat3(m00, m01, m02, m10, m11, m12, m20, m21, m22) {
    return [
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22]
    ];
}

function mat4(m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33) {
    return [
        [m00, m01, m02, m03],
        [m10, m11, m12, m13],
        [m20, m21, m22, m23],
        [m30, m31, m32, m33]
    ];
}

function lerp(P, Q, alpha) {
    if (typeof P === 'number' && typeof Q === 'number') {
        return alpha * Q + (1 - alpha) * P;
    } else {
        const result = [];
        for (let i = 0; i < P.length; i++) {
            result.push(alpha * Q[i] + (1 - alpha) * P[i]);
        }
        return result;
    }
}

function map_point(P, Q, A, B, X) {
    let alpha;
    if (typeof P === 'number' && typeof Q === 'number') {
        if (Math.abs(Q - P) < 0.00001) {
            alpha = 0;
        } else {
            alpha = (X - P) / (Q - P);
        }
    } else {
        if (Math.abs(Q[0] - P[0]) < 0.00001) {
            if (Q.length > 1 && Math.abs(Q[1] - P[1]) > 0.00001) {
                alpha = (X[1] - P[1]) / (Q[1] - P[1]);
            } else {
                alpha = 0;
            }
        } else {
            alpha = (X[0] - P[0]) / (Q[0] - P[0]);
        }
    }
    
    return lerp(A, B, alpha);
}

// Flatten an array of vec2/vec3/vec4 into a Float32Array
function flatten(v) {
    if (v.length === 0) return new Float32Array(0);
    if (typeof v[0] === 'number') return new Float32Array(v);

    var floats = [];
    for (var i = 0; i < v.length; i++) {
        for (var j = 0; j < v[i].length; j++) {
            floats.push(v[i][j]);
        }
    }
    return new Float32Array(floats);
}

// Dot product
function dot(u, v) {
    var sum = 0.0;
    for (var i = 0; i < u.length; i++) {
        sum += u[i] * v[i];
    }
    return sum;
}

// Cross product (vec3)
function cross(u, v) {
    return vec3(
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    );
}

// Length of a vector
function length(v) {
    return Math.sqrt(dot(v, v));
}

// Normalize a vector
function normalize(v) {
    var len = length(v);
    if (len < 0.00001) return v;
    var result = [];
    for (var i = 0; i < v.length; i++) {
        result.push(v[i] / len);
    }
    return result;
}

// Subtract two vectors
function subtract(u, v) {
    var result = [];
    for (var i = 0; i < u.length; i++) {
        result.push(u[i] - v[i]);
    }
    return result;
}

// Add two vectors
function add(u, v) {
    var result = [];
    for (var i = 0; i < u.length; i++) {
        result.push(u[i] + v[i]);
    }
    return result;
}

// Scale a vector
function scale(s, v) {
    var result = [];
    for (var i = 0; i < v.length; i++) {
        result.push(s * v[i]);
    }
    return result;
}

// Negate a vector
function negate(v) {
    return scale(-1, v);
}

// Identity matrix (4x4 column-major flat array for WebGL)
function identity() {
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ];
}

// Rotation about X axis (angle in degrees), column-major
function rotateX(angle) {
    var r = angle * Math.PI / 180.0;
    var c = Math.cos(r), s = Math.sin(r);
    return [
        1, 0, 0, 0,
        0, c, s, 0,
        0, -s, c, 0,
        0, 0, 0, 1
    ];
}

// Rotation about Y axis (angle in degrees), column-major
function rotateY(angle) {
    var r = angle * Math.PI / 180.0;
    var c = Math.cos(r), s = Math.sin(r);
    return [
        c, 0, -s, 0,
        0, 1, 0, 0,
        s, 0, c, 0,
        0, 0, 0, 1
    ];
}

// Rotation about Z axis (angle in degrees), column-major
function rotateZ(angle) {
    var r = angle * Math.PI / 180.0;
    var c = Math.cos(r), s = Math.sin(r);
    return [
        c, s, 0, 0,
        -s, c, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ];
}

// Multiply two 4x4 matrices (column-major flat arrays)
function mult(a, b) {
    var result = new Array(16);
    for (var i = 0; i < 4; i++) {
        for (var j = 0; j < 4; j++) {
            var sum = 0;
            for (var k = 0; k < 4; k++) {
                sum += a[i + k*4] * b[k + j*4];
            }
            result[i + j*4] = sum;
        }
    }
    return result;
}

// Scale matrix (column-major)
function scalem(sx, sy, sz) {
    return [
        sx, 0, 0, 0,
        0, sy, 0, 0,
        0, 0, sz, 0,
        0, 0, 0, 1
    ];
}

// Translation matrix (column-major)
function translate(tx, ty, tz) {
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        tx, ty, tz, 1
    ];
}

// Perspective projection (column-major)
function perspective(fovy, aspect, near, far) {
    var f = 1.0 / Math.tan(fovy * Math.PI / 360.0);
    var d = far - near;
    return [
        f/aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, -(near+far)/d, -1,
        0, 0, -2*near*far/d, 0
    ];
}
