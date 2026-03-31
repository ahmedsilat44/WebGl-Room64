// =============================================================================
// WEBGL DOOM64-STYLE 3D ROOM DEMO
// Uses: common.js (MV library), initShaders.js, webgl-utils.js
// Pattern from: Angel & Shreiner "Interactive Computer Graphics" 7th ed.
// =============================================================================

// gl      - The WebGLRenderingContext; the main handle for all GPU operations.
// program - The linked shader program (vertex + fragment shaders compiled together).
var gl;
var program;

// ---------------------------------------------------------------------------
// ATTRIBUTE & UNIFORM LOCATIONS
// ---------------------------------------------------------------------------
// Attributes are per-vertex inputs fed from buffers:
//   aPos  - vec3 : world-space position of each vertex
//   aNorm - vec3 : surface normal used for lighting calculations
//   aCol  - vec3 : RGB colour baked into the vertex (no textures)
//
// Uniforms are constant values for an entire draw call:
//   uMVP      - mat4 : combined Model-View-Projection matrix (transforms to clip space)
//   uModel    - mat4 : model matrix (object-to-world; identity here, geometry pre-placed)
//   uNM       - mat3 : normal matrix = transpose(inverse(model)) upper-left 3x3
//                      keeps normals correct even if model is scaled non-uniformly
//   uMode     - int  : shading mode flag (0=wireframe, 1=flat, 2=smooth/Phong)
//   uLightPos - vec3 : world-space position of the single point light
//   uEye      - vec3 : world-space camera position for specular highlights
var aPos, aNorm, aCol;
var uMVP, uModel, uNM, uMode, uLightPos, uEye;

// ---------------------------------------------------------------------------
// GEOMETRY ACCUMULATOR
// ---------------------------------------------------------------------------
// All geometry is collected into flat arrays and uploaded to the GPU in batches.
// WebGL with Uint16 indices can only address 65536 vertices per draw call, so
// geometry is split into "batches" of at most ~62000 vertices (gBase tracks count).
//
//   batches - array of objects, each holding GPU buffer handles for one draw call
//   gP      - flat float array: positions   [x0,y0,z0, x1,y1,z1, ...]
//   gN      - flat float array: normals     [nx0,ny0,nz0, ...]
//   gC      - flat float array: colours     [r0,g0,b0, ...]
//   gI      - flat int array:   triangle indices [i0,i1,i2, i3,i4,i5, ...]
//   gBase   - running count of vertices added to current batch (for index offsetting)
var batches = [];
var gP = [], gN = [], gC = [], gI = [], gBase = 0;

// ---------------------------------------------------------------------------
// COLOUR PALETTE
// ---------------------------------------------------------------------------
// Named RGB colour constants used throughout the scene.  Values are in [0,1].
// Using a palette keeps the colour scheme consistent and easy to tweak.
//   floor/ceil - dark stone surfaces
//   w1-w4      - four wall colour variants (south, north, east/west, alcove side)
//   col/cap    - column shaft and base/cap colours
//   acc        - bright red accent trim strips on walls
//   lava       - orange glow colour for the lava pit floor and walls
//   dark       - near-black for ceiling beams
//   trim       - warm highlight for floor dividers and column caps
//   step       - blood-red for lava pit step terrain
//   plat       - concrete grey for raised side platforms
//   barrel/lid - steel tones for the industrial barrels
var C = {
    floor:  [0.13, 0.12, 0.10],
    ceil:   [0.07, 0.07, 0.09],
    w1:     [0.24, 0.14, 0.09],
    w2:     [0.16, 0.16, 0.20],
    w3:     [0.19, 0.09, 0.07],
    w4:     [0.10, 0.15, 0.10],
    col:    [0.22, 0.15, 0.11],
    cap:    [0.30, 0.20, 0.13],
    acc:    [0.55, 0.08, 0.04],
    lava:   [0.72, 0.28, 0.01],
    dark:   [0.05, 0.05, 0.06],
    trim:   [0.33, 0.23, 0.16],
    step:   [0.18, 0.08, 0.06],
    plat:   [0.20, 0.18, 0.16],
    barrel: [0.35, 0.35, 0.38],
    lid:    [0.20, 0.20, 0.22]
};

// ---------------------------------------------------------------------------
// ROOM DIMENSIONS
// ---------------------------------------------------------------------------
// The room is a rectangular box centred at world origin (X=0, Z=0).
//   RoomWidth  - total width  (X axis) in world units
//   RoomDepth  - total depth  (Z axis) in world units
//   WallHeight - height of all walls and ceiling (Y axis)
//   halfWidth/halfDepth - half-extents used for symmetrical placement
var RoomWidth = 30, RoomDepth = 40, WallHeight = 6;
var halfWidth = 15, halfDepth = 20;

// ---------------------------------------------------------------------------
// CAMERA STATE
// ---------------------------------------------------------------------------
// All camera parameters are stored in a single object for easy access.
//   x, y, z      - world-space eye position  (z=17 means starting near front wall)
//   yaw          - horizontal look angle in RADIANS (left/right, around Y axis)
//   pitch        - vertical look angle in RADIANS   (up/down, around X axis)
//   roll         - bank/tilt angle in RADIANS        (around Z axis, Q/E keys)
//   fov          - vertical field-of-view in DEGREES (Arrow Up/Down to zoom)
//   near/far     - clipping plane distances; geometry outside is not drawn
//   speed        - movement speed in world units per second
//   viewShift    - horizontal frustum asymmetry; shifts the view without rotating
//                  (Z/X keys; similar to screen-space head-bob)
var cam = {
    x: 0,   y: 2.2,  z: 17,
    yaw: 0, pitch: 0, roll: 0,
    fov: 75, near: 0.15, far: 250,
    speed: 5, viewShift: 0
};

// ---------------------------------------------------------------------------
// MOVEMENT BOUNDS
// ---------------------------------------------------------------------------
// Hard clamps applied every frame so the camera cannot walk through walls.
// Values are inset slightly from the room edges (0.4 units) to fake collision.
//   minX/maxX - left and right walls
//   minZ/maxZ - south wall to far end of hallway
//   minY/maxY - floor level (eye height ~1.8 units) to just below ceiling
var BOUNDS = {
    minX: -halfWidth + 0.4,  maxX:  halfWidth - 0.4,
    minZ: -halfDepth + 0.4,  maxZ:  halfDepth + 18 - 0.4,
    minY:  1.6,              maxY:  WallHeight - 0.4
};

// ---------------------------------------------------------------------------
// INPUT & RUNTIME STATE
// ---------------------------------------------------------------------------
//   keys          - dictionary keyed by e.code (e.g. 'KeyW', 'ArrowLeft');
//                   value is true while the key is held down, false/absent otherwise
//   pointerLocked - true when the browser Pointer Lock API has captured the mouse;
//                   enables raw mouse-delta look without the cursor escaping
//   shadingMode   - current shading mode: 0=wireframe, 1=flat, 2=smooth (Phong)
//   lastTime      - DOMHighResTimeStamp of the previous frame, used to compute dt
//   fps/fpsAcc/fpsCnt - rolling FPS counter averaged over 0.5-second windows
var keys = {};
var pointerLocked = false;
var shadingMode = 1;
var lastTime = 0, fps = 0, fpsAcc = 0, fpsCnt = 0;

// ---------------------------------------------------------------------------
// frustum(l, r, b, t, n, f)
// ---------------------------------------------------------------------------
// Builds a 4x4 perspective projection matrix from explicit clip-plane edges.
// This is the raw form of gluFrustum / glFrustum from classic OpenGL.
//
// Unlike the symmetric perspective() in common.js (which takes fov + aspect),
// this version accepts asymmetric left/right/bottom/top edges, enabling the
// viewShift effect: the frustum can be shifted sideways without rotating the
// camera (useful for stereo rendering, head-bob, or off-centre projection).
//
// Parameters:
//   l, r  - left and right edges of the near-plane rectangle in view space
//   b, t  - bottom and top edges of the near-plane rectangle
//   n, f  - near and far clipping distances (must both be positive)
//
// Maths (column-major, matching WebGL convention used by common.js):
//   M[0]  = 2n/(r-l)        -- horizontal scale
//   M[5]  = 2n/(t-b)        -- vertical scale
//   M[8]  = (r+l)/(r-l)     -- horizontal skew (non-zero when l != -r)
//   M[9]  = (t+b)/(t-b)     -- vertical skew
//   M[10] = -(f+n)/(f-n)    -- depth mapping
//   M[11] = -1               -- perspective divide (w = -z)
//   M[14] = -2fn/(f-n)      -- depth translation
//
// Returns a flat 16-element array (column-major to match Float32Array layout).
function frustum(l, r, b, t, n, f) {
    var result = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0];
    result[0]  =  (2 * n) / (r - l);
    result[5]  =  (2 * n) / (t - b);
    result[8]  =  (r + l) / (r - l);
    result[9]  =  (t + b) / (t - b);
    result[10] = -(f + n) / (f - n);
    result[11] = -1.0;
    result[14] = -(2 * f * n) / (f - n);
    return result;
}

// ---------------------------------------------------------------------------
// mat3Normal(m)
// ---------------------------------------------------------------------------
// Extracts the upper-left 3x3 sub-matrix from a 4x4 matrix m.
// This 3x3 is passed to the shader as the "normal matrix" (uniform uNM).
//
// WHY A SEPARATE NORMAL MATRIX?
//   Normals are direction vectors, not positions.  If the model matrix contains
//   non-uniform scaling, transforming a normal by the same matrix would skew
//   it and break lighting.  The correct transform is:
//       n_world = transpose(inverse(modelMatrix)) * n_local
//   For our demo the model matrix is always identity(), so the normal matrix is
//   also identity and we skip the inverse/transpose.  But the infrastructure is
//   here so you can animate objects (e.g., rotating monsters) correctly later.
//
// Parameters:
//   m - flat 16-element array (column-major 4x4 matrix, same format as common.js)
//
// Returns a Float32Array of 9 elements ready to pass to gl.uniformMatrix3fv().
function mat3Normal(m) {
    return new Float32Array([
        m[0], m[1], m[2],
        m[4], m[5], m[6],
        m[8], m[9], m[10]
    ]);
}


// ---------------------------------------------------------------------------
// flushBatch()
// ---------------------------------------------------------------------------
// Uploads the current contents of the geometry accumulators (gP, gN, gC, gI)
// to the GPU as a set of static WebGL buffers, then resets the accumulators
// so the next batch can start fresh.
//
// WHY BATCHING?
//   WebGL with Uint16Array indices can only address 2^16 = 65536 unique vertices
//   per draw call.  The scene has more than that, so geometry is split into
//   separate "batches".  Each batch is independent: its own VBOs and index buffer.
//   addQuad() / addBox() call flushBatch() automatically when gBase nears 62000.
//
// BUFFERS CREATED PER BATCH:
//   posB  - ARRAY_BUFFER : vec3 positions  (3 floats × N vertices)
//   norB  - ARRAY_BUFFER : vec3 normals    (3 floats × N vertices)
//   colB  - ARRAY_BUFFER : vec3 colours    (3 floats × N vertices)
//   idxB  - ELEMENT_ARRAY_BUFFER : triangle indices (Uint16, 2 triangles/quad = 6 ints)
//   wIdxB - ELEMENT_ARRAY_BUFFER : unique-edge wireframe indices (gl.LINES pairs)
//
// WIREFRAME INDEX BUILDING:
//   Iterates over every triangle (i, i+1, i+2) and emits the three edges
//   (a,b), (b,c), (a,c).  A hash-set (edgeSet) prevents duplicate edges from
//   being drawn twice, giving a clean wireframe without over-bright seams.
//
// After upload all accumulator arrays are cleared and gBase is reset to 0.
function flushBatch() {
    if (gP.length === 0) return;

    var posB = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posB);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(gP), gl.STATIC_DRAW);

    var norB = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, norB);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(gN), gl.STATIC_DRAW);

    var colB = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colB);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(gC), gl.STATIC_DRAW);

    var idxB = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxB);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(gI), gl.STATIC_DRAW);

    // Build wireframe edge index
    var edgeSet = {};
    var wireArr = [];
    for (var i = 0; i < gI.length; i += 3) {
        var a = gI[i], b = gI[i+1], c = gI[i+2];
        var edges = [[a,b],[b,c],[a,c]];
        for (var j = 0; j < edges.length; j++) {
            var p = edges[j][0], q = edges[j][1];
            var key = Math.min(p,q) + '|' + Math.max(p,q);
            if (!edgeSet[key]) {
                edgeSet[key] = true;
                wireArr.push(p, q);
            }
        }
    }
    var wIdxB = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, wIdxB);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(wireArr), gl.STATIC_DRAW);

    batches.push({
        posB: posB, norB: norB, colB: colB,
        idxB: idxB,  cnt:  gI.length,
        wIdxB: wIdxB, wcnt: wireArr.length
    });

    gP = [];  gN = [];  gC = [];  gI = [];  gBase = 0;
}

// ---------------------------------------------------------------------------
// addQuad(p0, p1, p2, p3, col)
// ---------------------------------------------------------------------------
// The core geometry primitive.  Adds one planar quadrilateral (rectangle or
// arbitrary flat quad) to the current accumulator batch.
//
// Parameters:
//   p0, p1, p2, p3 - four corner positions as [x, y, z] arrays, in
//                    COUNTER-CLOCKWISE order when viewed from the front face.
//                    CCW winding is the WebGL default for front-face culling.
//   col            - RGB colour as [r, g, b] array (same colour for all 4 verts)
//
// HOW IT WORKS:
//   1. NORMAL CALCULATION
//      Two edge vectors are computed:
//        e1 = p1 - p0   (first edge)
//        e2 = p3 - p0   (second edge, to opposite corner)
//      Their cross product gives a vector perpendicular to the quad surface.
//      normalize() turns it into a unit vector.
//      All 4 vertices share this same flat normal, which produces flat shading.
//      In smooth-shading mode the shader applies Phong interpolation using these
//      per-vertex normals, which here produces the same result as flat shading
//      for planar polygons (normals are identical at every vertex).
//
//   2. INDEX GENERATION
//      A quad is split into 2 triangles:
//        Triangle 1: (gBase+0, gBase+1, gBase+2)
//        Triangle 2: (gBase+0, gBase+2, gBase+3)
//      This is the standard "fan" decomposition: both triangles share the
//      diagonal p0->p2.  gBase is advanced by 4 after each quad.
//
//   3. BATCH OVERFLOW CHECK
//      If adding 4 more vertices would exceed 62000, flushBatch() is called
//      first so the indices never overflow Uint16Array.
//
// Dependencies: subtract(), cross(), normalize() from common.js
function addQuad(p0, p1, p2, p3, col) {
    if (gBase + 4 > 62000) flushBatch();

    var e1 = subtract(p1, p0);
    var e2 = subtract(p3, p0);
    var n  = normalize(cross(e1, e2));

    var verts = [p0, p1, p2, p3];
    for (var i = 0; i < 4; i++) {
        gP.push(verts[i][0], verts[i][1], verts[i][2]);
        gN.push(n[0], n[1], n[2]);
        gC.push(col[0], col[1], col[2]);
    }
    gI.push(gBase, gBase+1, gBase+2,  gBase, gBase+2, gBase+3);
    gBase += 4;
}

// ---------------------------------------------------------------------------
// addBox(x, y, z, W, H, D, sideCol, topCol, botCol)
// ---------------------------------------------------------------------------
// Builds an axis-aligned rectangular prism (cuboid) from 6 addQuad() calls,
// one per face, each with an outward-pointing normal.
//
// Parameters:
//   x, y, z          - position of the minimum corner (bottom-left-back)
//   W, H, D          - width (X), height (Y), depth (Z) extents
//   sideCol          - RGB colour for the 4 vertical side faces
//   topCol (optional)- RGB colour for the top (+Y) face; defaults to sideCol
//   botCol (optional)- RGB colour for the bottom (-Y) face; defaults to sideCol
//
// FACE ORDER & NORMALS (outward-facing CCW winding):
//   Bottom (-Y) : normal pointing down  -- floor of columns, platform bases
//   Top    (+Y) : normal pointing up    -- tops of columns, platform surfaces
//   Front  (+Z) : normal pointing +Z    -- towards viewer at scene start
//   Back   (-Z) : normal pointing -Z
//   Left   (-X) : normal pointing -X
//   Right  (+X) : normal pointing +X
//
// USAGE IN SCENE:
//   - Rooms walls are NOT built with addBox (they are individual addQuad calls
//     so only the visible inner face is drawn -- no overdraw waste).
//   - addBox is used for SOLID objects where all sides may be visible:
//     columns, ceiling beams, floor dividers, raised platforms, lava steps.
function addBox(x, y, z, W, H, D, sideCol, topCol, botCol) {
    var sc = sideCol;
    var tc = topCol || sc;
    var bc = botCol || sc;
    var x1 = x+W, y1 = y+H, z1 = z+D;

    addQuad([x,y,z1],[x,y,z],  [x1,y,z],  [x1,y,z1], bc);   // bottom (-Y)
    addQuad([x,y1,z],[x,y1,z1],[x1,y1,z1],[x1,y1,z], tc);    // top    (+Y)
    addQuad([x,y,z1],[x1,y,z1],[x1,y1,z1],[x,y1,z1], sc);    // front  (+Z)
    addQuad([x1,y,z],[x,y,z],  [x,y1,z],  [x1,y1,z], sc);    // back   (-Z)
    addQuad([x,y,z], [x,y,z1], [x,y1,z1], [x,y1,z],  sc);    // left   (-X)
    addQuad([x1,y,z1],[x1,y,z],[x1,y1,z],[x1,y1,z1], sc);    // right  (+X)
}

// ---------------------------------------------------------------------------
// addCylinder(x, y, z, radius, height, sides, sideCol, capCol)
// ---------------------------------------------------------------------------
// Approximates a cylinder using a prism with `sides` rectangular side faces
// and `sides` triangular top/bottom cap fans.
//
// Parameters:
//   x, y, z   - centre of the cylinder BASE on the XZ plane (not the centroid)
//   radius    - distance from axis to the outer edge (barrel ~0.6 units)
//   height    - vertical extent; top is at y + height
//   sides     - number of rectangular side panels; higher = rounder (10 for barrels)
//   sideCol   - RGB colour for the vertical side quads
//   capCol    - RGB colour for the top and bottom cap faces
//
// HOW IT WORKS:
//   The circle is divided into `sides` equal arc segments.
//   For each segment i, two angles are computed:
//     a1 = i     * (2π / sides)   -- start angle
//     a2 = (i+1) * (2π / sides)   -- end angle
//   Cartesian coordinates at each angle give two vertices on the circumference:
//     (x + cos(a) * radius, y_bottom/top, z + sin(a) * radius)
//
//   THREE FACES PER SEGMENT:
//     1. Side quad  : (x1,bot,z1) -> (x2,bot,z2) -> (x2,top,z2) -> (x1,top,z1)
//        This is one rectangular panel of the curved surface.
//     2. Top cap    : centre (x,topY,z) + two arc vertices
//        NOTE: The cap uses a degenerate quad (p2==p3) which degenerates to a
//        triangle.  For a more correct fan the cap would use gl.TRIANGLE_FAN,
//        but here addQuad handles it fine at low side counts.
//     3. Bottom cap : same as top, mirrored and wound oppositely for outward normal.
//
// LIMITATION: At sides=10 the cylinder looks slightly faceted (like a barrel
// should).  Increase sides to 16+ for smoother pipes or columns if needed.
function addCylinder(x, y, z, radius, height, sides, sideCol, capCol) {
    var topY = y + height;
    var botY = y;
    var step = (Math.PI * 2) / sides;

    for (var i = 0; i < sides; i++) {
        var a1 = i * step;
        var a2 = (i + 1) * step;
        var x1 = x + Math.cos(a1) * radius;
        var z1 = z + Math.sin(a1) * radius;
        var x2 = x + Math.cos(a2) * radius;
        var z2 = z + Math.sin(a2) * radius;

        addQuad([x1,botY,z1],[x2,botY,z2],[x2,topY,z2],[x1,topY,z1], sideCol);
        addQuad([x,topY,z],[x1,topY,z1],[x2,topY,z2],[x2,topY,z2], capCol);
        addQuad([x,botY,z],[x2,botY,z2],[x1,botY,z1],[x1,botY,z1], capCol);
    }
}


// ---------------------------------------------------------------------------
// buildScene()
// ---------------------------------------------------------------------------
// Constructs the entire 3D scene by calling addQuad(), addBox(), and
// addCylinder() in world-space coordinates.  All geometry is static (built
// once at startup) and uploaded to GPU buffers via flushBatch().
//
// SCENE LAYOUT (top-down, Z axis points toward viewer):
//
//  Z = +38  ┌──────[ hallway end wall ]──────┐
//           │          Hallway (7×18)         │
//  Z = +20  ╠═════[ hallway entrance ]════════╣
//           │    North wall panels             │
//           │  [ column ]        [ column ]    │
//  Z = +6   │         (lava pit area)          │
//           │  [ column ]        [ column ]    │
//  Z = -4   │                                  │
//           │  [ column ]        [ column ]    │
//  Z = -14  │                                  │
//           │  [ plat ]          [ plat ]      │
//           ╠═[ alcove left ]══[ alcove right ]═╣
//  Z = -20  └────────[ south wall ]─────────────┘
//                  (with alcove notch)
//
// CONSTRUCTION ORDER:
//   1.  Floor & Ceiling quads (single large quads)
//   2.  South / North / West / East walls
//   3.  Accent trim strips (thin quads offset 0.01 from wall to avoid z-fighting)
//   4.  Ceiling beams     (addBox in a loop along Z)
//   5.  Floor lane dividers (thin addBox strips)
//   6.  Eight columns     (each = base box + shaft box + cap box)
//   7.  Lava pit          (terrain steps + lava floor quad + pit walls)
//   8.  Raised side platforms (addBox)
//   9.  South alcove      (6 face quads extending the south wall outward)
//   10. North hallway     (6 face quads + accent trim + flanking wall panels)
//   11. flushBatch()      -- uploads everything above to one GPU batch
//   12. Industrial barrels (random placement with lava-pit avoidance check)
//   13. flushBatch()      -- uploads barrels as a second GPU batch
//
// Z-FIGHTING:
//   Overlapping coplanar quads (accent strips, thin dividers) are offset by
//   0.01 world units perpendicular to the wall so the GPU depth test can
//   distinguish them.  This is the standard simple fix before considering
//   glPolygonOffset.
function buildScene() {
    var T0 = 0.85, T1 = 1.15;

    // Floor
    addQuad([-halfWidth,0, halfDepth],[ halfWidth,0, halfDepth],
            [ halfWidth,0,-halfDepth],[-halfWidth,0,-halfDepth], C.floor);

    // Ceiling
    addQuad([-halfWidth,WallHeight,-halfDepth],[ halfWidth,WallHeight,-halfDepth],
            [ halfWidth,WallHeight, halfDepth],[-halfWidth,WallHeight, halfDepth], C.ceil);

    // South wall (z = -halfDepth)
    addQuad([ halfWidth,0,-halfDepth],[-halfWidth,0,-halfDepth],
            [-halfWidth,WallHeight,-halfDepth],[ halfWidth,WallHeight,-halfDepth], C.w1);

    // North wall (z = +halfDepth)
    addQuad([-halfWidth,0, halfDepth],[ halfWidth,0, halfDepth],
            [ halfWidth,WallHeight, halfDepth],[-halfWidth,WallHeight, halfDepth], C.w2);

    // West wall (x = -halfWidth)
    addQuad([-halfWidth,0,-halfDepth],[-halfWidth,0, halfDepth],
            [-halfWidth,WallHeight, halfDepth],[-halfWidth,WallHeight,-halfDepth], C.w3);

    // East wall (x = +halfWidth)
    addQuad([ halfWidth,0, halfDepth],[ halfWidth,0,-halfDepth],
            [ halfWidth,WallHeight,-halfDepth],[ halfWidth,WallHeight, halfDepth], C.w3);

    // Accent trim strips at y ~ 1
    addQuad([ halfWidth,T0,-halfDepth+0.01],[-halfWidth,T0,-halfDepth+0.01],
            [-halfWidth,T1,-halfDepth+0.01],[ halfWidth,T1,-halfDepth+0.01], C.acc);
    addQuad([-halfWidth,T0, halfDepth-0.01],[ halfWidth,T0, halfDepth-0.01],
            [ halfWidth,T1, halfDepth-0.01],[-halfWidth,T1, halfDepth-0.01], C.acc);
    addQuad([-halfWidth+0.01,T0,-halfDepth],[-halfWidth+0.01,T0, halfDepth],
            [-halfWidth+0.01,T1, halfDepth],[-halfWidth+0.01,T1,-halfDepth], C.acc);
    addQuad([ halfWidth-0.01,T0, halfDepth],[ halfWidth-0.01,T0,-halfDepth],
            [ halfWidth-0.01,T1,-halfDepth],[ halfWidth-0.01,T1, halfDepth], C.acc);

    // Ceiling beams
    for (var bz = -16; bz <= 16; bz += 8) {
        addBox(-halfWidth, WallHeight-0.32, bz-0.2, RoomWidth, 0.31, 0.4, C.dark);
    }

    // Floor lane dividers
    for (var i = -2; i <= 2; i++) {
        addBox(i*7 - 0.05, 0.005, -halfDepth, 0.1, 0.07, RoomDepth, C.trim);
    }

    // Columns (8, placed symmetrically)
    var columnPositions = [
        [-9, -14], [ 9, -14],
        [-9,  -4], [ 9,  -4],
        [-9,   6], [ 9,   6],
        [-5,  15], [ 5,  15]
    ];
    for (var ci = 0; ci < columnPositions.length; ci++) {
        var cx = columnPositions[ci][0], cz = columnPositions[ci][1];
        addBox(cx-0.9,  0,              cz-0.9,  1.8, 0.28, 1.8, C.cap, C.trim, C.cap);
        addBox(cx-0.55, 0.28,           cz-0.55, 1.1, WallHeight-0.56, 1.1, C.col);
        addBox(cx-0.9,  WallHeight-0.28, cz-0.9, 1.8, 0.28, 1.8, C.cap, C.trim, C.cap);
    }

    // Lava pit with stepped terrain
    var LX = 0, LZ = 5;
    addBox(LX-5.5, -0.28, LZ-5.5, 11, 0.275, 11, C.w3, C.step, C.w3);
    addBox(LX-3.5, -0.52, LZ-3.5,  7, 0.24,  7,  C.step, C.acc, C.step);

    // Lava floor
    addQuad([LX-2.4,-0.52,LZ+2.4],[LX+2.4,-0.52,LZ+2.4],
            [LX+2.4,-0.52,LZ-2.4],[LX-2.4,-0.52,LZ-2.4], C.lava);

    // Pit walls
    addQuad([LX-2.4,-0.52,LZ-2.4],[LX-2.4,-0.52,LZ+2.4],
            [LX-2.4,0,LZ+2.4],[LX-2.4,0,LZ-2.4], C.lava);
    addQuad([LX+2.4,-0.52,LZ+2.4],[LX+2.4,-0.52,LZ-2.4],
            [LX+2.4,0,LZ-2.4],[LX+2.4,0,LZ+2.4], C.lava);
    addQuad([LX-2.4,-0.52,LZ+2.4],[LX+2.4,-0.52,LZ+2.4],
            [LX+2.4,0,LZ+2.4],[LX-2.4,0,LZ+2.4], C.acc);
    addQuad([LX+2.4,-0.52,LZ-2.4],[LX-2.4,-0.52,LZ-2.4],
            [LX-2.4,0,LZ-2.4],[LX+2.4,0,LZ-2.4], C.acc);

    // Raised platforms on the sides
    addBox(-halfWidth+0.01, 0.005, -16, 5, 0.45, 7, C.plat, C.floor, C.w2);
    addBox( halfWidth-5.01, 0.005, -16, 5, 0.45, 7, C.plat, C.floor, C.w2);

    // South alcove
    var ax = 3.5, az = 1.5;
    addQuad([-ax,0,-halfDepth-az],[ ax,0,-halfDepth-az],
            [ ax,0,-halfDepth],[-ax,0,-halfDepth], C.floor);
    addQuad([-ax,WallHeight,-halfDepth],[ ax,WallHeight,-halfDepth],
            [ ax,WallHeight,-halfDepth-az],[-ax,WallHeight,-halfDepth-az], C.ceil);
    addQuad([ ax,0,-halfDepth-az],[-ax,0,-halfDepth-az],
            [-ax,WallHeight,-halfDepth-az],[ ax,WallHeight,-halfDepth-az], C.w1);
    addQuad([-ax,0,-halfDepth],[-ax,0,-halfDepth-az],
            [-ax,WallHeight,-halfDepth-az],[-ax,WallHeight,-halfDepth], C.w4);
    addQuad([ ax,0,-halfDepth-az],[ ax,0,-halfDepth],
            [ ax,WallHeight,-halfDepth],[ ax,WallHeight,-halfDepth-az], C.w4);
    addQuad([ halfWidth,0,-halfDepth],[ ax,0,-halfDepth],
            [ ax,WallHeight,-halfDepth],[ halfWidth,WallHeight,-halfDepth], C.w1);
    addQuad([-ax,0,-halfDepth],[-halfWidth,0,-halfDepth],
            [-halfWidth,WallHeight,-halfDepth],[-ax,WallHeight,-halfDepth], C.w1);

    // Hallway extending north
    var hallWidth = 7, hallDepth = 18;
    var halfHall = hallWidth / 2;

    addQuad([-halfHall,0,halfDepth+hallDepth],[ halfHall,0,halfDepth+hallDepth],
            [ halfHall,0,halfDepth],[-halfHall,0,halfDepth], C.floor);
    addQuad([-halfHall,WallHeight,halfDepth],[ halfHall,WallHeight,halfDepth],
            [ halfHall,WallHeight,halfDepth+hallDepth],[-halfHall,WallHeight,halfDepth+hallDepth], C.ceil);
    addQuad([-halfHall,0,halfDepth],[-halfHall,0,halfDepth+hallDepth],
            [-halfHall,WallHeight,halfDepth+hallDepth],[-halfHall,WallHeight,halfDepth], C.w4);
    addQuad([ halfHall,0,halfDepth+hallDepth],[ halfHall,0,halfDepth],
            [ halfHall,WallHeight,halfDepth],[ halfHall,WallHeight,halfDepth+hallDepth], C.w4);
    addQuad([ halfHall,0,halfDepth+hallDepth],[-halfHall,0,halfDepth+hallDepth],
            [-halfHall,WallHeight,halfDepth+hallDepth],[ halfHall,WallHeight,halfDepth+hallDepth], C.w1);
    addQuad([-halfHall+0.01,T0,halfDepth],[-halfHall+0.01,T0,halfDepth+hallDepth],
            [-halfHall+0.01,T1,halfDepth+hallDepth],[-halfHall+0.01,T1,halfDepth], C.acc);
    addQuad([ halfHall-0.01,T0,halfDepth+hallDepth],[ halfHall-0.01,T0,halfDepth],
            [ halfHall-0.01,T1,halfDepth],[ halfHall-0.01,T1,halfDepth+hallDepth], C.acc);
    // North wall panels flanking hallway
    addQuad([-halfWidth,0,halfDepth],[-halfHall,0,halfDepth],
            [-halfHall,WallHeight,halfDepth],[-halfWidth,WallHeight,halfDepth], C.w2);
    addQuad([ halfHall,0,halfDepth],[ halfWidth,0,halfDepth],
            [ halfWidth,WallHeight,halfDepth],[ halfHall,WallHeight,halfDepth], C.w2);

    flushBatch();

    // Industrial barrels (random placement, avoids lava pit)
    var barrelCount = 3;
    var placed = 0;
    var attempts = 0;
    while (placed < barrelCount && attempts < 100) {
        var bx = (Math.random() - 0.5) * (RoomWidth - 6);
        var bz = (Math.random() - 0.5) * (RoomDepth - 10);
        var dx = bx - 0, dz = bz - 5;
        var distToLava = Math.sqrt(dx * dx + dz * dz);
        if (distToLava >= 6) {
            addCylinder(bx, 0, bz, 0.6, 1.6, 10, C.barrel, C.lid);
            placed++;
        }
        attempts++;
    }

    flushBatch();
}


// ---------------------------------------------------------------------------
// setSh(m)
// ---------------------------------------------------------------------------
// Switches the active shading mode and updates the HUD buttons to reflect it.
//
// Parameters:
//   m - integer mode index:
//         0 = WIREFRAME  : draws only triangle edges (gl.LINES, wIdxB buffer)
//         1 = FLAT       : solid triangles, uniform colour per face, basic diffuse
//         2 = SMOOTH     : Phong shading interpolated per-pixel in fragment shader
//
// HOW IT WORKS:
//   1. Stores m in the global shadingMode (read by drawBatch() and render()).
//   2. Updates CSS classes on all '.sbtn' buttons: the active one gets class 'on'.
//   3. Updates the text of the '#hshd' HUD element.
//
// The shader itself reads uMode (the GPU copy of shadingMode set in render())
// and branches inside the fragment shader:
//   uMode == 0 : outputs a fixed dark colour (wireframe edges don't need shading)
//   uMode == 1 : Lambert diffuse  = max(0, dot(N, L)) × colour × lightColour
//   uMode == 2 : Phong  = ambient + diffuse + specular (Blinn-Phong half-vector)
function setSh(m) {
    shadingMode = m;
    var btns = document.querySelectorAll('.sbtn');
    for (var i = 0; i < btns.length; i++) {
        if (i === m) btns[i].className = 'sbtn on';
        else btns[i].className = 'sbtn';
    }
    document.getElementById('hshd').textContent =
        'SHADING: ' + ['WIREFRAME', 'FLAT', 'SMOOTH'][m];
}

// ---------------------------------------------------------------------------
// update(dt)
// ---------------------------------------------------------------------------
// Reads the keyboard state (keys{}) and integrates camera position & orientation
// forward by dt seconds.  Called every frame from render() before drawing.
//
// Parameters:
//   dt - delta-time in seconds since last frame (clamped to 0.05 to prevent
//        large jumps after tab switches or long frames)
//
// MOVEMENT:
//   The camera moves in a FIRST-PERSON style: forward/back/strafe are relative
//   to the current yaw angle, NOT world axes.  This is done by decomposing
//   yaw into a forward vector (fx, fz) and a right vector (rx, rz):
//     forward = (sin(yaw), 0, -cos(yaw))   -- points in look direction on XZ plane
//     right   = (cos(yaw), 0,  sin(yaw))   -- perpendicular, 90° clockwise
//   W/S add/subtract the forward vector; A/D add/subtract the right vector.
//   R/F move straight up/down (no head-bob, no gravity -- fly-mode).
//   ShiftLeft/ShiftRight: 2.5× speed multiplier (sprint).
//
// LOOK:
//   ArrowLeft/Right: yaw (turn left/right) at 1 radian/second.
//   Mouse: yaw and pitch are set externally in mousemove handler (not in update).
//   Q/E: roll (bank) the camera at 1 rad/s -- purely visual, no movement effect.
//
// FOV & CLIPPING:
//   Arrow Up/Down: zoom in/out by adjusting cam.fov ±25°/s (clamped 20–130°).
//   [ / ] keys: shrink near plane / expand far plane.
//
// VIEW SHIFT:
//   Z/X: nudge cam.viewShift ±0.4/s -- passed to frustum() to skew the projection.
//
// BOUNDS CLAMPING:
//   After all movement, cam.x/y/z are clamped to BOUNDS values to fake wall
//   collision.  There is no real physics -- objects can still be clipped through
//   at high speed.  For assignment purposes this is sufficient.
function update(dt) {
    var spd = cam.speed * ((keys.ShiftLeft || keys.ShiftRight) ? 2.5 : 1.0);

    var cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    var fx = sy,  fz = -cy;
    var rx = cy,  rz = sy;

    if (keys.KeyW) { cam.x += fx*spd*dt;  cam.z += fz*spd*dt; }
    if (keys.KeyS) { cam.x -= fx*spd*dt;  cam.z -= fz*spd*dt; }
    if (keys.KeyA) { cam.x -= rx*spd*dt;  cam.z -= rz*spd*dt; }
    if (keys.KeyD) { cam.x += rx*spd*dt;  cam.z += rz*spd*dt; }
    if (keys.KeyR) cam.y += spd * dt;
    if (keys.KeyF) cam.y -= spd * dt;

    if (keys.ArrowLeft)  cam.yaw -= dt;
    if (keys.ArrowRight) cam.yaw += dt;
    if (keys.KeyQ) cam.roll -= dt;
    if (keys.KeyE) cam.roll += dt;

    if (keys.ArrowUp)   cam.fov = Math.max(20,  cam.fov - 25*dt);
    if (keys.ArrowDown) cam.fov = Math.min(130, cam.fov + 25*dt);

    if (keys.BracketLeft)  cam.near = Math.max(0.05, cam.near - 0.3*dt);
    if (keys.BracketRight) cam.far  = Math.min(500,  cam.far  + 20*dt);

    if (keys.KeyZ) cam.viewShift -= 0.4 * dt;
    if (keys.KeyX) cam.viewShift += 0.4 * dt;

    if (keys.Comma)  cam.speed = Math.max(1,  cam.speed - 3*dt);
    if (keys.Period) cam.speed = Math.min(25, cam.speed + 3*dt);

    cam.x = Math.max(BOUNDS.minX, Math.min(BOUNDS.maxX, cam.x));
    cam.z = Math.max(BOUNDS.minZ, Math.min(BOUNDS.maxZ, cam.z));
    cam.y = Math.max(BOUNDS.minY, Math.min(BOUNDS.maxY, cam.y));
}

// ---------------------------------------------------------------------------
// drawBatch(b)
// ---------------------------------------------------------------------------
// Issues a single WebGL draw call for one previously-uploaded geometry batch.
//
// Parameters:
//   b - a batch object from the batches[] array, containing:
//         posB  - position VBO handle
//         norB  - normal VBO handle
//         colB  - colour VBO handle
//         idxB  - triangle index buffer handle
//         cnt   - number of triangle indices
//         wIdxB - wireframe edge index buffer handle
//         wcnt  - number of wireframe line indices
//
// PROCEDURE:
//   1. Bind each attribute buffer and call vertexAttribPointer() to describe
//      the layout: 3 floats, tightly packed (stride=0), starting at offset=0.
//      enableVertexAttribArray() activates the attribute so the shader reads it.
//   2. If shadingMode === 0 (wireframe), bind wIdxB and draw as gl.LINES.
//      Otherwise bind idxB and draw as gl.TRIANGLES.
//
// NOTE on vertexAttribPointer parameters:
//   (index, size=3, type=FLOAT, normalized=false, stride=0, offset=0)
//   stride=0 means "tightly packed" (WebGL infers the stride from size×sizeof(type)).
//   normalized=false because we're already storing actual float values, not
//   packed int8 or int16 that need range expansion.
function drawBatch(b) {
    gl.bindBuffer(gl.ARRAY_BUFFER, b.posB);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, b.norB);
    gl.enableVertexAttribArray(aNorm);
    gl.vertexAttribPointer(aNorm, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, b.colB);
    gl.enableVertexAttribArray(aCol);
    gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);

    if (shadingMode === 0) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.wIdxB);
        gl.drawElements(gl.LINES, b.wcnt, gl.UNSIGNED_SHORT, 0);
    } else {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.idxB);
        gl.drawElements(gl.TRIANGLES, b.cnt, gl.UNSIGNED_SHORT, 0);
    }
}

// ---------------------------------------------------------------------------
// render(now)
// ---------------------------------------------------------------------------
// The main animation loop, registered with requestAnimFrame() (from webgl-utils.js,
// which wraps requestAnimationFrame with a cross-browser fallback).
// Called automatically by the browser once per display refresh (typically 60 Hz).
//
// Parameters:
//   now - DOMHighResTimeStamp in milliseconds provided by the browser
//
// FRAME SEQUENCE:
//   1.  Re-register self: requestAnimFrame(render) keeps the loop going.
//   2.  Compute dt = (now - lastTime) / 1000 clamped to 0.05 s.
//   3.  Accumulate FPS counter; update display every 0.5 s.
//   4.  Call update(dt) to move the camera based on held keys.
//
// VIEW MATRIX CONSTRUCTION:
//   In OpenGL/WebGL the view matrix transforms world-space vertices into
//   camera/eye space.  It is the inverse of the camera's world transform.
//   Here we build it as a product of rotation and translation matrices
//   (all from common.js):
//     T   = translate(-cam.x, -cam.y, -cam.z)   move world so camera is at origin
//     Ry  = rotateY(yaw)                         undo yaw rotation
//     Rx  = rotateX(pitch)                       undo pitch
//     Rz  = rotateZ(roll)                        undo roll
//     view = Rz * Rx * Ry * T
//   NOTE: common.js rotateX/Y/Z take degrees, so angles are converted:
//     degrees = radians × (180 / π)
//
// PROJECTION MATRIX:
//   Uses the custom frustum() function (not common.js perspective()) to
//   support the asymmetric viewShift feature.  Half-height at near plane:
//     tHalf = near × tan(fov/2)
//   Right/left edges incorporate viewShift:
//     rEdge =  tHalf × aspect + viewShift × tHalf
//     lEdge = -tHalf × aspect + viewShift × tHalf
//
// UNIFORMS SENT TO SHADER EACH FRAME:
//   uMVP      - combined proj × view (model is identity here)
//   uModel    - identity (objects are pre-placed in world space)
//   uNM       - normal matrix (upper-left 3x3 of model)
//   uMode     - shading mode integer
//   uLightPos - fixed at (0, 4.5, 0) -- centre of room, above floor
//   uEye      - current camera position for specular calculation
//
// HUD UPDATE:
//   After drawing, updates text elements in the HTML overlay with live values
//   for position, angles (converted to degrees with ×57.3 ≈ 180/π), FOV,
//   clipping planes, speed, and FPS.
function render(now) {
    requestAnimFrame(render);

    var dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    fpsAcc += dt;  fpsCnt++;
    if (fpsAcc >= 0.5) { fps = Math.round(fpsCnt / fpsAcc); fpsAcc = 0; fpsCnt = 0; }

    update(dt);

    // View matrix: View = Rz * Rx * Ry * T(-pos)
    var RAD = 180.0 / Math.PI;
    var T   = translate(-cam.x, -cam.y, -cam.z);
    var Ry  = rotateY(cam.yaw * RAD);
    var Rx  = rotateX(cam.pitch * RAD);
    var Rz  = rotateZ(cam.roll * RAD);
    var view = mult(Rz, mult(Rx, mult(Ry, T)));

    // Projection (asymmetric frustum for view-bound control)
    var asp   = gl.canvas.width / gl.canvas.height;
    var fovR  = cam.fov * Math.PI / 180;
    var tHalf = cam.near * Math.tan(fovR / 2);
    var rEdge =  tHalf * asp + cam.viewShift * tHalf;
    var lEdge = -tHalf * asp + cam.viewShift * tHalf;
    var proj  = frustum(lEdge, rEdge, -tHalf, tHalf, cam.near, cam.far);

    var MVP = mult(proj, view);
    var MDL = identity();
    var NM  = mat3Normal(MDL);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.uniformMatrix4fv(uMVP,   false, new Float32Array(MVP));
    gl.uniformMatrix4fv(uModel, false, new Float32Array(MDL));
    gl.uniformMatrix3fv(uNM,    false, NM);
    gl.uniform1i(uMode, shadingMode);
    gl.uniform3f(uLightPos, 0, 4.5, 0);
    gl.uniform3f(uEye, cam.x, cam.y, cam.z);

    for (var i = 0; i < batches.length; i++) {
        drawBatch(batches[i]);
    }

    // HUD update
    document.getElementById('hpos').textContent =
        'POS: ' + cam.x.toFixed(1) + ', ' + cam.y.toFixed(1) + ', ' + cam.z.toFixed(1);
    document.getElementById('hang').textContent =
        'YAW: ' + (cam.yaw*57.3).toFixed(0) + '\u00B0  PITCH: ' + (cam.pitch*57.3).toFixed(0) + '\u00B0  ROLL: ' + (cam.roll*57.3).toFixed(0) + '\u00B0';
    document.getElementById('hfov').textContent =
        'FOV: ' + cam.fov.toFixed(0) + '\u00B0  NEAR: ' + cam.near.toFixed(2) + '  FAR: ' + cam.far.toFixed(0);
    document.getElementById('hspd').textContent =
        'SPEED: ' + cam.speed.toFixed(1);
    document.getElementById('hfps').textContent =
        'FPS: ' + fps;
}

// ---------------------------------------------------------------------------
// startGame()
// ---------------------------------------------------------------------------
// Called when the user clicks the "ENTER" overlay button.
// Hides the start-screen overlay, requests Pointer Lock on the canvas so
// mouse movement controls the camera without the cursor escaping the window,
// and kicks off the render loop with the first requestAnimFrame() call.
//
// Pointer Lock API:
//   canvas.requestPointerLock() -- browser may prompt the user to allow it.
//   Once granted, mousemove events carry e.movementX / e.movementY (raw delta
//   in pixels) instead of absolute coordinates.  The pointerlockchange listener
//   set up in init() keeps pointerLocked in sync so update() knows whether
//   to ignore mouse input.
function startGame() {
    document.getElementById('overlay').style.display = 'none';
    gl.canvas.requestPointerLock();
    requestAnimFrame(render);
}


// ---------------------------------------------------------------------------
// init()  [called as window.onload]
// ---------------------------------------------------------------------------
// Entry point -- runs once after the HTML page has fully loaded.
// Follows the standard Angel & Shreiner assignment pattern:
//   1. Get canvas by ID and initialise WebGL context via WebGLUtils.setupWebGL()
//   2. Set up a resizeCanvas() handler so the viewport fills the window at all sizes
//   3. Configure global GL state (clear colour, depth test)
//   4. Compile & link shaders with initShaders() (reads <script> tags by ID)
//   5. Look up all attribute and uniform locations
//   6. Build scene geometry (buildScene())
//   7. Attach all input event listeners (keyboard, pointer lock, mouse)
//
// WebGLUtils.setupWebGL(canvas):
//   From webgl-utils.js -- wraps canvas.getContext('webgl') with browser prefix
//   fallbacks (webkit, moz) and prints an alert if WebGL is not supported.
//
// initShaders(gl, vertexShaderId, fragmentShaderId):
//   From initShaders.js -- reads the innerHTML of <script> tags with those IDs,
//   compiles them as VERTEX_SHADER / FRAGMENT_SHADER, links them into a program,
//   and returns the program handle.  Logs errors to the console if compilation fails.
//
// EVENT LISTENERS:
//   keydown/keyup      : sets keys[e.code] true/false for held-key detection
//   canvas click       : requests pointer lock (must originate from user gesture)
//   pointerlockchange  : syncs pointerLocked flag
//   mousemove          : updates cam.yaw and cam.pitch from raw mouse delta
//                        sensitivity = 0.0022 radians/pixel (adjustable)
//                        pitch clamped to ±1.55 rad (≈ ±89°) to prevent gimbal flip
//   keydown (1/2/3)    : shading mode shortcuts
window.onload = function init() {
    var canvas = document.getElementById("gl-canvas");
    gl = WebGLUtils.setupWebGL(canvas);
    if (!gl) { alert("WebGL isn't available"); return; }

    function resizeCanvas() {
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
        gl.viewport(0, 0, canvas.width, canvas.height);
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    gl.clearColor(0.01, 0.01, 0.015, 1.0);
    gl.enable(gl.DEPTH_TEST);

    program = initShaders(gl, "vertex-shader", "fragment-shader");
    gl.useProgram(program);

    // Attribute locations
    aPos  = gl.getAttribLocation(program, 'aPos');
    aNorm = gl.getAttribLocation(program, 'aNorm');
    aCol  = gl.getAttribLocation(program, 'aCol');

    // Uniform locations
    uMVP      = gl.getUniformLocation(program, 'uMVP');
    uModel    = gl.getUniformLocation(program, 'uModel');
    uNM       = gl.getUniformLocation(program, 'uNM');
    uMode     = gl.getUniformLocation(program, 'uMode');
    uLightPos = gl.getUniformLocation(program, 'uLightPos');
    uEye      = gl.getUniformLocation(program, 'uEye');

    // Build all geometry
    buildScene();

    // Input handlers
    document.addEventListener('keydown', function(e) { keys[e.code] = true; });
    document.addEventListener('keyup',   function(e) { keys[e.code] = false; });

    canvas.addEventListener('click', function() {
        if (!pointerLocked) canvas.requestPointerLock();
    });
    document.addEventListener('pointerlockchange', function() {
        pointerLocked = !!document.pointerLockElement;
    });
    document.addEventListener('mousemove', function(e) {
        if (!pointerLocked) return;
        cam.yaw   += e.movementX * 0.0022;
        cam.pitch += e.movementY * 0.0022;
        cam.pitch  = Math.max(-1.55, Math.min(1.55, cam.pitch));
    });

    // Keyboard shortcuts for shading modes
    document.addEventListener('keydown', function(e) {
        if (e.code === 'Digit1') setSh(0);
        if (e.code === 'Digit2') setSh(1);
        if (e.code === 'Digit3') setSh(2);
    });
};
