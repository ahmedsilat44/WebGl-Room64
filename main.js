"use strict";

// ─── Canvas & WebGL ──────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
if (!gl) { alert('WebGL not supported in this browser.'); }

function resizeCanvas() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  gl.viewport(0, 0, canvas.width, canvas.height);
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// ─── GLSL Shader Sources ─────────────────────────────────────────────────────
const VS_SRC = `
  attribute vec3 aPos;
  attribute vec3 aNorm;
  attribute vec3 aCol;

  uniform mat4 uMVP;
  uniform mat4 uModel;
  uniform mat3 uNM;
  uniform int  uMode;

  varying vec3 vCol;
  varying vec3 vNorm;
  varying vec3 vFrag;
  varying vec3 vFlatCol;

  void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFrag       = (uModel * vec4(aPos, 1.0)).xyz;
    vNorm       = normalize(uNM * aNorm);
    vCol        = aCol;

    // Flat shading: diffuse computed per-vertex using face normal
    // (all 4 verts of a quad share the same normal, so output is constant across the face)
    vec3 L   = normalize(vec3(0.4, 1.0, 0.5));
    float d  = max(dot(vNorm, L), 0.0);
    vFlatCol = aCol * (0.22 + 0.78 * d);
  }
`;

const FS_SRC = `
  precision mediump float;

  varying vec3 vCol;
  varying vec3 vNorm;
  varying vec3 vFrag;
  varying vec3 vFlatCol;

  uniform int  uMode;
  uniform vec3 uLightPos;
  uniform vec3 uEye;

  void main() {
    if (uMode == 0) {
      // Wireframe: slightly brightened face colour for contrast
      gl_FragColor = vec4(min(vCol * 1.6, vec3(1.0)), 1.0);

    } else if (uMode == 1) {
      // Flat shading (Gouraud with constant normal per face)
      gl_FragColor = vec4(vFlatCol, 1.0);

    } else {
      // Phong shading
      vec3 N   = normalize(vNorm);
      vec3 L   = normalize(uLightPos - vFrag);
      vec3 V   = normalize(uEye - vFrag);
      vec3 R   = reflect(-L, N);
      float dif = max(dot(N, L), 0.0);
      float spe = pow(max(dot(V, R), 0.0), 48.0) * 0.45;
      vec3 col  = vCol * 0.22 + vCol * dif + vec3(spe);
      gl_FragColor = vec4(col, 1.0);
    }
  }
`;

// ─── Compile & Link Shaders ──────────────────────────────────────────────────
function compileShader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    console.error('Shader compile error:', gl.getShaderInfoLog(s));
  return s;
}

const program = gl.createProgram();
gl.attachShader(program, compileShader(gl.VERTEX_SHADER,   VS_SRC));
gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, FS_SRC));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS))
  console.error('Program link error:', gl.getProgramInfoLog(program));
gl.useProgram(program);

// Attribute / uniform locations
const A_POS  = gl.getAttribLocation(program,  'aPos');
const A_NORM = gl.getAttribLocation(program,  'aNorm');
const A_COL  = gl.getAttribLocation(program,  'aCol');
const U_MVP  = gl.getUniformLocation(program, 'uMVP');
const U_MDL  = gl.getUniformLocation(program, 'uModel');
const U_NM   = gl.getUniformLocation(program, 'uNM');
const U_MODE = gl.getUniformLocation(program, 'uMode');
const U_LP   = gl.getUniformLocation(program, 'uLightPos');
const U_EYE  = gl.getUniformLocation(program, 'uEye');

// ─── Matrix Library (column-major, matching WebGL / GLSL convention) ─────────
//
// Column-major layout: m[col * 4 + row]
//
//   m[0]  m[4]  m[8]  m[12]
//   m[1]  m[5]  m[9]  m[13]
//   m[2]  m[6]  m[10] m[14]
//   m[3]  m[7]  m[11] m[15]
//

function mat4()         { return new Float32Array(16); }
function mat4Identity() { const m = mat4(); m[0]=m[5]=m[10]=m[15]=1; return m; }

/** C = A * B  (column-major matrix multiply) */
function mat4Mul(A, B) {
  const C = mat4();
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += A[k*4 + row] * B[col*4 + k];
      C[col*4 + row] = sum;
    }
  }
  return C;
}

/** Translation matrix */
function mat4Translate(x, y, z) {
  const m = mat4Identity();
  m[12] = x;  m[13] = y;  m[14] = z;
  return m;
}

/**
 * Rotation around Y axis (yaw).
 * Standard right-hand RotY:
 *   col0 = [ cos, 0, -sin, 0 ]
 *   col1 = [  0,  1,   0,  0 ]
 *   col2 = [ sin, 0,  cos, 0 ]
 *   col3 = [  0,  0,   0,  1 ]
 */
function mat4RotY(a) {
  const m = mat4Identity();
  const c = Math.cos(a), s = Math.sin(a);
  m[0]  =  c;   // row0, col0
  m[2]  = -s;   // row2, col0
  m[8]  =  s;   // row0, col2
  m[10] =  c;   // row2, col2
  return m;
}

/**
 * Rotation around X axis (pitch).
 *   col0 = [ 1,  0,   0,  0 ]
 *   col1 = [ 0, cos, sin,  0 ]
 *   col2 = [ 0,-sin, cos,  0 ]
 *   col3 = [ 0,  0,   0,  1 ]
 */
function mat4RotX(a) {
  const m = mat4Identity();
  const c = Math.cos(a), s = Math.sin(a);
  m[5]  =  c;   // row1, col1
  m[6]  =  s;   // row2, col1
  m[9]  = -s;   // row1, col2
  m[10] =  c;   // row2, col2
  return m;
}

/**
 * Rotation around Z axis (roll).
 *   col0 = [ cos, sin, 0, 0 ]
 *   col1 = [-sin, cos, 0, 0 ]
 *   col2 = [  0,   0,  1, 0 ]
 *   col3 = [  0,   0,  0, 1 ]
 */
function mat4RotZ(a) {
  const m = mat4Identity();
  const c = Math.cos(a), s = Math.sin(a);
  m[0]  =  c;   // row0, col0
  m[1]  =  s;   // row1, col0
  m[4]  = -s;   // row0, col1
  m[5]  =  c;   // row1, col1
  return m;
}

/**
 * Asymmetric frustum projection matrix.
 * Allows left/right view-bound shift for the view-bound requirement.
 * l,r = left/right clip planes, b,t = bottom/top, n,f = near/far.
 */
function mat4Frustum(l, r, b, t, n, f) {
  const m = mat4();
  m[0]  =  (2 * n) / (r - l);
  m[5]  =  (2 * n) / (t - b);
  m[8]  =  (r + l) / (r - l);        // col2, row0
  m[9]  =  (t + b) / (t - b);        // col2, row1
  m[10] = -(f + n) / (f - n);        // col2, row2
  m[11] = -1.0;                       // col2, row3 — perspective divide
  m[14] = -(2 * f * n) / (f - n);    // col3, row2
  return m;
}

/** Extract upper-left 3×3 from a mat4 (used as normal matrix for identity model). */
function mat3FromMat4(m) {
  return new Float32Array([
    m[0], m[1], m[2],
    m[4], m[5], m[6],
    m[8], m[9], m[10],
  ]);
}

// ─── Vec3 Helpers ────────────────────────────────────────────────────────────
function cross(a, b) {
  return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ];
}
function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/l, v[1]/l, v[2]/l];
}

// ─── Geometry Accumulator ────────────────────────────────────────────────────
//
// Geometry is accumulated in CPU arrays and uploaded to GPU in "batches".
// A new batch is started whenever the running vertex count approaches 65535
// (the Uint16 index limit).  Each batch becomes its own set of VBOs.
//
const batches = [];
let gP = [], gN = [], gC = [], gI = [], gBase = 0;

function flushBatch() {
  if (gP.length === 0) return;

  // Upload position / normal / colour / index buffers
  const posB = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posB);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(gP), gl.STATIC_DRAW);

  const norB = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, norB);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(gN), gl.STATIC_DRAW);

  const colB = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colB);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(gC), gl.STATIC_DRAW);

  const idxB = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxB);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(gI), gl.STATIC_DRAW);

  // Build unique-edge list for wireframe mode
  const edgeSet = new Set();
  const wireArr = [];
  for (let i = 0; i < gI.length; i += 3) {
    const a = gI[i], b = gI[i+1], c = gI[i+2];
    [[a,b],[b,c],[a,c]].forEach(([p, q]) => {
      const key = Math.min(p,q) + '|' + Math.max(p,q);
      if (!edgeSet.has(key)) { edgeSet.add(key); wireArr.push(p, q); }
    });
  }
  const wIdxB = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, wIdxB);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(wireArr), gl.STATIC_DRAW);

  batches.push({
    posB, norB, colB,
    idxB,  cnt:  gI.length,
    wIdxB, wcnt: wireArr.length,
  });

  // Reset accumulators for next batch
  gP = [];  gN = [];  gC = [];  gI = [];  gBase = 0;
}

/**
 * Add a quad defined by 4 vertices (p0..p3).
 * Vertices must be ordered counter-clockwise as seen from the visible side.
 * The face normal is computed automatically from cross(p1-p0, p3-p0).
 */
function addQuad(p0, p1, p2, p3, col) {
  if (gBase + 4 > 62000) flushBatch();    // guard against Uint16 overflow

  const e1 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
  const e2 = [p3[0]-p0[0], p3[1]-p0[1], p3[2]-p0[2]];
  const n  = normalize(cross(e1, e2));

  [p0, p1, p2, p3].forEach(p => {
    gP.push(p[0], p[1], p[2]);
    gN.push(n[0], n[1], n[2]);
    gC.push(col[0], col[1], col[2]);
  });
  gI.push(gBase, gBase+1, gBase+2,  gBase, gBase+2, gBase+3);
  gBase += 4;
}

/**
 * Add an axis-aligned box.
 * All 6 faces have normals pointing outward; vertices are CCW from outside.
 * sideCol  — colour for all 4 vertical faces
 * topCol   — colour for the top face  (defaults to sideCol)
 * botCol   — colour for the bottom face (defaults to sideCol)
 */
function addBox(x, y, z, W, H, D, sideCol, topCol, botCol) {
  const sc = sideCol;
  const tc = topCol || sc;
  const bc = botCol || sc;
  const x1 = x+W, y1 = y+H, z1 = z+D;

  addQuad([x,y,z1],[x,y,z],  [x1,y,z],  [x1,y,z1], bc);  // bottom (-Y)
  addQuad([x,y1,z],[x,y1,z1],[x1,y1,z1],[x1,y1,z], tc);  // top    (+Y)
  addQuad([x,y,z1],[x1,y,z1],[x1,y1,z1],[x,y1,z1], sc);  // front  (+Z)
  addQuad([x1,y,z],[x,y,z],  [x,y1,z],  [x1,y1,z], sc);  // back   (-Z)
  addQuad([x,y,z], [x,y,z1], [x,y1,z1], [x,y1,z],  sc);  // left   (-X)
  addQuad([x1,y,z1],[x1,y,z],[x1,y1,z],[x1,y1,z1], sc);  // right  (+X)
}

// cylinder code for barrels
/**
 * Adds a cylinder (barrel) to the geometry.
 * sides: number of vertical faces (e.g., 8 or 12)
 */
function addCylinder(x, y, z, radius, height, sides, sideCol, capCol) {
  const topY = y + height;
  const botY = y;
  const step = (Math.PI * 2) / sides;

  for (let i = 0; i < sides; i++) {
    const a1 = i * step;
    const a2 = (i + 1) * step;

    // Corner coordinates for the side face
    const x1 = x + Math.cos(a1) * radius;
    const z1 = z + Math.sin(a1) * radius;
    const x2 = x + Math.cos(a2) * radius;
    const z2 = z + Math.sin(a2) * radius;

    // 1. Side panel (Quad)
    addQuad(
      [x1, botY, z1], 
      [x2, botY, z2], 
      [x2, topY, z2], 
      [x1, topY, z1], 
      sideCol
    );

    // 2. Top Cap (Triangle - we use addQuad but collapse one side)
    // Connecting [x, topY, z] center to the two outer points
    addQuad(
      [x, topY, z],
      [x1, topY, z1],
      [x2, topY, z2],
      [x2, topY, z2], // Duplicate point to make a triangle
      capCol
    );

    // 3. Bottom Cap (Triangle)
    addQuad(
      [x, botY, z],
      [x2, botY, z2],
      [x1, botY, z1],
      [x1, botY, z1],
      capCol
    );
  }
}
// ─── Colour Palette ──────────────────────────────────────────────────────────
const C = {
  floor: [0.13, 0.12, 0.10],
  ceil:  [0.07, 0.07, 0.09],
  w1:    [0.24, 0.14, 0.09],   // south wall  — brownish
  w2:    [0.16, 0.16, 0.20],   // north wall  — grayish
  w3:    [0.19, 0.09, 0.07],   // east/west   — dark red
  w4:    [0.10, 0.15, 0.10],   // hallway     — dark green
  col:   [0.22, 0.15, 0.11],   // column shaft
  cap:   [0.30, 0.20, 0.13],   // column cap/base
  acc:   [0.55, 0.08, 0.04],   // accent trim — blood red
  lava:  [0.72, 0.28, 0.01],   // lava orange
  dark:  [0.05, 0.05, 0.06],   // ceiling beams
  trim:  [0.33, 0.23, 0.16],   // floor strips
  step:  [0.18, 0.08, 0.06],   // lava pit steps
  plat:  [0.20, 0.18, 0.16],   // side platforms
  barrel: [0.35, 0.35, 0.38],  // barrels in hallway
  lid:    [0.20, 0.20, 0.22],  // barrel top/bottom caps
};

// ─────────────────────────────────────────────────────────────────────────────
//  OBJECT 1 — Main Room
//
//  Coordinate system: X = right, Y = up, Z = toward viewer (right-hand).
//  Room bounds: X ∈ [-15, +15],  Y ∈ [0, 6],  Z ∈ [-20, +20].
//  Camera spawns at [0, 2.2, 17] with yaw = 0  (looking toward −Z / south).
// ─────────────────────────────────────────────────────────────────────────────
const RoomWidth = 30, RoomDepth = 40, WallHeight = 6;
const halfWidth = 15, halfDepth = 20;   // half-extents

// Floor — normal +Y (CCW from above)
addQuad([-halfWidth,0, halfDepth],[ halfWidth,0, halfDepth],[ halfWidth,0,-halfDepth],[-halfWidth,0,-halfDepth],  C.floor);

// Ceiling — normal −Y (CCW from below)
addQuad([-halfWidth,WallHeight,-halfDepth],[ halfWidth,WallHeight,-halfDepth],[ halfWidth,WallHeight, halfDepth],[-halfWidth,WallHeight, halfDepth],  C.ceil);

// South wall (z = −halfDepth) — normal +Z, CCW from inside
addQuad([ halfWidth,0,-halfDepth],[-halfWidth,0,-halfDepth],[-halfWidth,WallHeight,-halfDepth],[ halfWidth,WallHeight,-halfDepth],  C.w1);

// North wall (z = +halfDepth) — normal −Z, CCW from inside
addQuad([-halfWidth,0, halfDepth],[ halfWidth,0, halfDepth],[ halfWidth,WallHeight, halfDepth],[-halfWidth,WallHeight, halfDepth],  C.w2);

// West wall (x = −halfWidth) — normal +X, CCW from inside
addQuad([-halfWidth,0,-halfDepth],[-halfWidth,0, halfDepth],[-halfWidth,WallHeight, halfDepth],[-halfWidth,WallHeight,-halfDepth],  C.w3);

// East wall (x = +halfWidth) — normal −X, CCW from inside
addQuad([ halfWidth,0, halfDepth],[ halfWidth,0,-halfDepth],[ halfWidth,WallHeight,-halfDepth],[ halfWidth,WallHeight, halfDepth],  C.w3);

// Horizontal accent trim strips at y ≈ 1 on each wall (offset 0.01 inward to avoid Z-fighting)
const T0 = 0.85, T1 = 1.15;
addQuad([ halfWidth,T0,-halfDepth+0.01],[-halfWidth,T0,-halfDepth+0.01],[-halfWidth,T1,-halfDepth+0.01],[ halfWidth,T1,-halfDepth+0.01],  C.acc);   // south
addQuad([-halfWidth,T0, halfDepth-0.01],[ halfWidth,T0, halfDepth-0.01],[ halfWidth,T1, halfDepth-0.01],[-halfWidth,T1, halfDepth-0.01],  C.acc);   // north
addQuad([-halfWidth+0.01,T0,-halfDepth],[-halfWidth+0.01,T0, halfDepth],[-halfWidth+0.01,T1, halfDepth],[-halfWidth+0.01,T1,-halfDepth],  C.acc);   // west
addQuad([ halfWidth-0.01,T0, halfDepth],[ halfWidth-0.01,T0,-halfDepth],[ halfWidth-0.01,T1,-halfDepth],[ halfWidth-0.01,T1, halfDepth],  C.acc);   // east


// Ceiling beams — cross the full room width every 8 units in Z
for (let bz = -16; bz <= 16; bz += 8) {
  addBox(-halfWidth, WallHeight-0.32, bz-0.2,  RoomWidth, 0.31, 0.4,  C.dark);
}

// Floor lane dividers — thin raised strips running the room length
for (let i = -2; i <= 2; i++) {
  addBox(i*7 - 0.05, 0.005, -halfDepth,  0.1, 0.07, RoomDepth,  C.trim);
}

// ─────────────────────────────────────────────────────────────────────────────
//  OBJECT 2 — Procedural Columns (8, placed symmetrically in two rows)
// ─────────────────────────────────────────────────────────────────────────────
const columnPositions = [
  [-9, -14], [ 9, -14],
  [-9,  -4], [ 9,  -4],
  [-9,   6], [ 9,   6],
  [-5,  15], [ 5,  15],
];

columnPositions.forEach(([cx, cz]) => {
  addBox(cx-0.9,  0,        cz-0.9,  1.8, 0.28,      1.8,  C.cap, C.trim, C.cap);  // base pedestal
  addBox(cx-0.55, 0.28,     cz-0.55, 1.1, WallHeight-0.56,   1.1,  C.col);         // shaft
  addBox(cx-0.9,  WallHeight-0.28,  cz-0.9,  1.8, 0.28,      1.8,  C.cap, C.trim, C.cap);  // capital
});

// ─────────────────────────────────────────────────────────────────────────────
//  OBJECT 3 — Lava Pit with Stepped Terrain
// ─────────────────────────────────────────────────────────────────────────────
const LX = 0, LZ = 5;

addBox(LX-5.5, -0.28, LZ-5.5,  11, 0.275, 11,  C.w3,  C.step, C.w3);  // outer step ring
addBox(LX-3.5, -0.52, LZ-3.5,   7, 0.24,  7,  C.step, C.acc, C.step);  // inner step ring

// Lava floor — normal +Y (CCW from above)
addQuad(
  [LX-2.4, -0.52, LZ+2.4],
  [LX+2.4, -0.52, LZ+2.4],
  [LX+2.4, -0.52, LZ-2.4],
  [LX-2.4, -0.52, LZ-2.4],
  C.lava
);

// Pit walls — normals pointing inward (toward pit centre)
addQuad([LX-2.4,-0.52,LZ-2.4],[LX-2.4,-0.52,LZ+2.4],[LX-2.4,0,LZ+2.4],[LX-2.4,0,LZ-2.4], C.lava); // west  +X
addQuad([LX+2.4,-0.52,LZ+2.4],[LX+2.4,-0.52,LZ-2.4],[LX+2.4,0,LZ-2.4],[LX+2.4,0,LZ+2.4], C.lava); // east  -X
addQuad([LX-2.4,-0.52,LZ+2.4],[LX+2.4,-0.52,LZ+2.4],[LX+2.4,0,LZ+2.4],[LX-2.4,0,LZ+2.4], C.acc);  // north -Z
addQuad([LX+2.4,-0.52,LZ-2.4],[LX-2.4,-0.52,LZ-2.4],[LX-2.4,0,LZ-2.4],[LX+2.4,0,LZ-2.4], C.acc);  // south +Z

// ─── Extra Decorative Geometry ───────────────────────────────────────────────

// Raised platforms on the left and right sides
addBox(-halfWidth+0.01, 0.005, -16,  5, 0.45, 7,  C.plat, C.floor, C.w2);
addBox( halfWidth-5.01, 0.005, -16,  5, 0.45, 7,  C.plat, C.floor, C.w2);

// South alcove — a recess behind the south wall
{
  const ax = 3.5, az = 1.5;
  addQuad([-ax,0,-halfDepth-az],[ ax,0,-halfDepth-az],[ ax,0,-halfDepth],[-ax,0,-halfDepth], C.floor); // alcove floor
  addQuad([-ax,WallHeight,-halfDepth],[ ax,WallHeight,-halfDepth],[ ax,WallHeight,-halfDepth-az],[-ax,WallHeight,-halfDepth-az], C.ceil);  // alcove ceiling
  addQuad([ ax,0,-halfDepth-az],[-ax,0,-halfDepth-az],[-ax,WallHeight,-halfDepth-az],[ ax,WallHeight,-halfDepth-az], C.w1); // back wall
  addQuad([-ax,0,-halfDepth],[-ax,0,-halfDepth-az],[-ax,WallHeight,-halfDepth-az],[-ax,WallHeight,-halfDepth], C.w4); // left side
  addQuad([ ax,0,-halfDepth-az],[ ax,0,-halfDepth],[ ax,WallHeight,-halfDepth],[ ax,WallHeight,-halfDepth-az], C.w4); // right side
  // Remaining south wall panels flanking the doorway
  addQuad([ halfWidth,0,-halfDepth],[ ax,0,-halfDepth],[ ax,WallHeight,-halfDepth],[ halfWidth,WallHeight,-halfDepth], C.w1);  // right panel
  addQuad([-ax,0,-halfDepth],[-halfWidth,0,-halfDepth],[-halfWidth,WallHeight,-halfDepth],[-ax,WallHeight,-halfDepth], C.w1);  // left panel
}

// Hallway extending north out of the main room
{
  const hallWidth = 7, hallDepth = 18;
  const halfHall = hallWidth / 2;

  addQuad([-halfHall,0,halfDepth+hallDepth],[ halfHall,0,halfDepth+hallDepth],[ halfHall,0,halfDepth],[-halfHall,0,halfDepth], C.floor);                // floor
  addQuad([-halfHall,WallHeight,halfDepth],[ halfHall,WallHeight,halfDepth],[ halfHall,WallHeight,halfDepth+hallDepth],[-halfHall,WallHeight,halfDepth+hallDepth], C.ceil); // ceiling
  addQuad([-halfHall,0,halfDepth],[-halfHall,0,halfDepth+hallDepth],[-halfHall,WallHeight,halfDepth+hallDepth],[-halfHall,WallHeight,halfDepth], C.w4);    // west wall
  addQuad([ halfHall,0,halfDepth+hallDepth],[ halfHall,0,halfDepth],[ halfHall,WallHeight,halfDepth],[ halfHall,WallHeight,halfDepth+hallDepth], C.w4);    // east wall
  addQuad([ halfHall,0,halfDepth+hallDepth],[-halfHall,0,halfDepth+hallDepth],[-halfHall,WallHeight,halfDepth+hallDepth],[ halfHall,WallHeight,halfDepth+hallDepth], C.w1); // end cap
  addQuad([-halfHall+0.01,T0,halfDepth],[-halfHall+0.01,T0,halfDepth+hallDepth],[-halfHall+0.01,T1,halfDepth+hallDepth],[-halfHall+0.01,T1,halfDepth], C.acc); // west trim
  addQuad([ halfHall-0.01,T0,halfDepth+hallDepth],[ halfHall-0.01,T0,halfDepth],[ halfHall-0.01,T1,halfDepth],[ halfHall-0.01,T1,halfDepth+hallDepth], C.acc); // east trim
  // North wall panels flanking the hallway opening
  addQuad([-halfWidth,0,halfDepth],[-halfHall,0,halfDepth],[-halfHall,WallHeight,halfDepth],[-halfWidth,WallHeight,halfDepth], C.w2);  // left panel
  addQuad([ halfHall,0,halfDepth],[ halfWidth,0,halfDepth],[ halfWidth,WallHeight,halfDepth],[ halfHall,WallHeight,halfDepth], C.w2);  // right panel
}

// Upload all remaining geometry to GPU
flushBatch();

// ─── Camera State ────────────────────────────────────────────────────────────
const cam = {
  x: 0,   y: 2.2,  z: 17,   // spawn inside room, near north wall
  yaw:   0,                   // 0 = looking toward −Z (south, into the room)
  pitch: 0,
  roll:  0,
  fov:   75,                  // vertical field of view in degrees
  near:  0.15,
  far:   250,
  speed: 5,
  viewShift: 0,               // horizontal frustum offset for left/right bound control
};

// Camera movement bounds (keep player inside the world)
const BOUNDS = {
  minX: -halfWidth + 0.4,   maxX:  halfWidth - 0.4,
  minZ: -halfDepth + 0.4,   maxZ:  halfDepth + 18 - 0.4,  // south wall to hallway end
  minY:  1.6,               maxY:  WallHeight - 0.4,
};

// ─── OBJECT 4 — Industrial Barrels ───────────────────────────────────────────
const barrelCount = 3;
for (let i = 0; i < barrelCount; i++) {
  // Random position within the main room bounds
  const bx = (Math.random() - 0.5) * (RoomWidth - 6); 
  const bz = (Math.random() - 0.5) * (RoomDepth - 10);
  
  // Don't place barrels inside the lava pit (LX=0, LZ=5, radius~5)
  const distToLava = Math.hypot(bx - LX, bz - LZ);
  if (distToLava < 6) { i--; continue; } 

  addCylinder(bx, 0, bz, 0.6, 1.6, 10, C.barrel, C.lid);
}

// Barrels are appended after the previous flush; upload this final batch too.
flushBatch();

// ─── Input Handling ──────────────────────────────────────────────────────────
const keys = {};
document.addEventListener('keydown', e => { keys[e.code] = true;  });
document.addEventListener('keyup',   e => { keys[e.code] = false; });

// Pointer lock for mouse look
let pointerLocked = false;
canvas.addEventListener('click', () => {
  if (!pointerLocked) canvas.requestPointerLock();
});
document.addEventListener('pointerlockchange', () => {
  pointerLocked = !!document.pointerLockElement;
});
document.addEventListener('mousemove', e => {
  if (!pointerLocked) return;
  cam.yaw   += e.movementX * 0.0022;
  cam.pitch += e.movementY * 0.0022;
  cam.pitch  = Math.max(-1.55, Math.min(1.55, cam.pitch));
});

// ─── Shading Mode ────────────────────────────────────────────────────────────
let shadingMode = 1;   // 0 = wireframe, 1 = flat, 2 = smooth (Phong)

function setSh(m) {
  shadingMode = m;
  document.querySelectorAll('.sbtn').forEach((b, i) => b.classList.toggle('on', i === m));
  document.getElementById('hshd').textContent =
    'SHADING: ' + ['WIREFRAME', 'FLAT', 'SMOOTH'][m];
}

// Keyboard shortcuts 1/2/3
document.addEventListener('keydown', e => {
  if (e.code === 'Digit1') setSh(0);
  if (e.code === 'Digit2') setSh(1);
  if (e.code === 'Digit3') setSh(2);
});

// ─── Update (physics / input) ────────────────────────────────────────────────
function update(dt) {
  const spd = cam.speed * (keys.ShiftLeft || keys.ShiftRight ? 2.5 : 1.0);

  // Horizontal movement vectors derived from yaw
  // yaw = 0  →  forward = −Z,  right = +X
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const fx = sy,  fz = -cy;   // forward
  const rx =  cy,  rz = sy;   // right (strafe)

  if (keys.KeyW) { cam.x += fx*spd*dt;  cam.z += fz*spd*dt; }
  if (keys.KeyS) { cam.x -= fx*spd*dt;  cam.z -= fz*spd*dt; }
  if (keys.KeyA) { cam.x -= rx*spd*dt;  cam.z -= rz*spd*dt; }
  if (keys.KeyD) { cam.x += rx*spd*dt;  cam.z += rz*spd*dt; }
  if (keys.KeyR)  cam.y += spd * dt;    // fly up
  if (keys.KeyF)  cam.y -= spd * dt;    // fly down

  // Yaw — arrow keys (pitch handled via mouse; arrow up/down controls FOV)
  if (keys.ArrowLeft)  cam.yaw -= dt;
  if (keys.ArrowRight) cam.yaw += dt;

  // Roll
  if (keys.KeyQ) cam.roll -= dt;
  if (keys.KeyE) cam.roll += dt;

  // FOV (top / bottom view bounds)
  if (keys.ArrowUp)   cam.fov = Math.max(20,  cam.fov - 25*dt);
  if (keys.ArrowDown) cam.fov = Math.min(130, cam.fov + 25*dt);

  // Near / far plane adjustment
  if (keys.BracketLeft)  cam.near = Math.max(0.05, cam.near - 0.3*dt);
  if (keys.BracketRight) cam.far  = Math.min(500,  cam.far  + 20*dt);

  // Left / right view-bound shift (asymmetric frustum)
  if (keys.KeyZ) cam.viewShift -= 0.4 * dt;
  if (keys.KeyX) cam.viewShift += 0.4 * dt;

  // Speed
  if (keys.Comma)  cam.speed = Math.max(1,  cam.speed - 3*dt);
  if (keys.Period) cam.speed = Math.min(25, cam.speed + 3*dt);

  // Clamp to world bounds
  cam.x = Math.max(BOUNDS.minX, Math.min(BOUNDS.maxX, cam.x));
  cam.z = Math.max(BOUNDS.minZ, Math.min(BOUNDS.maxZ, cam.z));
  cam.y = Math.max(BOUNDS.minY, Math.min(BOUNDS.maxY, cam.y));
}

// ─── Render (draw one batch of VBOs) ─────────────────────────────────────────
function drawBatch(b) {
  gl.bindBuffer(gl.ARRAY_BUFFER, b.posB);
  gl.enableVertexAttribArray(A_POS);
  gl.vertexAttribPointer(A_POS, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, b.norB);
  gl.enableVertexAttribArray(A_NORM);
  gl.vertexAttribPointer(A_NORM, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, b.colB);
  gl.enableVertexAttribArray(A_COL);
  gl.vertexAttribPointer(A_COL, 3, gl.FLOAT, false, 0, 0);

  if (shadingMode === 0) {
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.wIdxB);
    gl.drawElements(gl.LINES, b.wcnt, gl.UNSIGNED_SHORT, 0);
  } else {
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.idxB);
    gl.drawElements(gl.TRIANGLES, b.cnt, gl.UNSIGNED_SHORT, 0);
  }
}

// ─── Main Loop ───────────────────────────────────────────────────────────────
let lastTime = 0, fps = 0, fpsAcc = 0, fpsCnt = 0;

function frame(now) {
  requestAnimationFrame(frame);

  const dt = Math.min((now - lastTime) / 1000, 0.05);
  lastTime = now;
  fpsAcc += dt;  fpsCnt++;
  if (fpsAcc >= 0.5) { fps = Math.round(fpsCnt / fpsAcc); fpsAcc = 0; fpsCnt = 0; }

  update(dt);

  // ── View matrix  (View = Rz * Rx * Ry * T(-pos)) ─────────────────────────
  // Translation is applied first (moves world so camera is at origin),
  // then camera rotations are applied in order: yaw → pitch → roll.
  const T    = mat4Translate(-cam.x, -cam.y, -cam.z);
  const Ry   = mat4RotY(cam.yaw);
  const Rx   = mat4RotX(cam.pitch);
  const Rz   = mat4RotZ(cam.roll);
  const view = mat4Mul(Rz, mat4Mul(Rx, mat4Mul(Ry, T)));

  // ── Projection matrix (asymmetric frustum for view-bound control) ─────────
  const asp    = canvas.width / canvas.height;
  const fovR   = cam.fov * Math.PI / 180;
  const tHalf  = cam.near * Math.tan(fovR / 2);
  const rEdge  =  tHalf * asp + cam.viewShift * tHalf;
  const lEdge  = -tHalf * asp + cam.viewShift * tHalf;
  const proj   = mat4Frustum(lEdge, rEdge, -tHalf, tHalf, cam.near, cam.far);

  const MVP = mat4Mul(proj, view);
  const MDL = mat4Identity();
  const NM  = mat3FromMat4(MDL);

  // ── WebGL draw call ───────────────────────────────────────────────────────
  gl.clearColor(0.01, 0.01, 0.015, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  // No face culling — keeps all faces visible regardless of winding edge cases

  gl.uniformMatrix4fv(U_MVP,  false, MVP);
  gl.uniformMatrix4fv(U_MDL,  false, MDL);
  gl.uniformMatrix3fv(U_NM,   false, NM);
  gl.uniform1i(U_MODE, shadingMode);
  gl.uniform3f(U_LP,  0, 4.5, 0);
  gl.uniform3f(U_EYE, cam.x, cam.y, cam.z);

  batches.forEach(drawBatch);

  // ── HUD update ────────────────────────────────────────────────────────────
  document.getElementById('hpos').textContent =
    `POS: ${cam.x.toFixed(1)}, ${cam.y.toFixed(1)}, ${cam.z.toFixed(1)}`;
  document.getElementById('hang').textContent =
    `YAW: ${(cam.yaw*57.3).toFixed(0)}°  PITCH: ${(cam.pitch*57.3).toFixed(0)}°  ROLL: ${(cam.roll*57.3).toFixed(0)}°`;
  document.getElementById('hfov').textContent =
    `FOV: ${cam.fov.toFixed(0)}°  NEAR: ${cam.near.toFixed(2)}  FAR: ${cam.far.toFixed(0)}`;
  document.getElementById('hspd').textContent =
    `SPEED: ${cam.speed.toFixed(1)}`;
  document.getElementById('hfps').textContent =
    `FPS: ${fps}`;
}

// ─── Entry Point ─────────────────────────────────────────────────────────────
function startGame() {
  document.getElementById('overlay').style.display = 'none';
  canvas.requestPointerLock();
  requestAnimationFrame(frame);
}
