var gl;
var program;

// Attribute & uniform locations
var aPos, aNorm, aCol;
var uMVP, uModel, uNM, uMode, uLightPos, uEye;

// Geometry accumulator (batched to stay under Uint16 index limit)
var batches = [];
var gP = [], gN = [], gC = [], gI = [], gBase = 0;

// Colour palette
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

// Room dimensions
var RoomWidth = 30, RoomDepth = 40, WallHeight = 6;
var halfWidth = 15, halfDepth = 20;

// Camera state
var cam = {
    x: 0,   y: 2.2,  z: 17,
    yaw: 0, pitch: 0, roll: 0,
    fov: 75, near: 0.15, far: 250,
    speed: 5, viewShift: 0
};

// Movement bounds
var BOUNDS = {
    minX: -halfWidth + 0.4,  maxX:  halfWidth - 0.4,
    minZ: -halfDepth + 0.4,  maxZ:  halfDepth + 18 - 0.4,
    minY:  1.6,              maxY:  WallHeight - 0.4
};

// Input state
var keys = {};
var pointerLocked = false;
var shadingMode = 1;
var lastTime = 0, fps = 0, fpsAcc = 0, fpsCnt = 0;

// Asymmetric frustum (not in common.js)
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

// Extract upper-left 3x3 normal matrix from a 4x4
function mat3Normal(m) {
    return new Float32Array([
        m[0], m[1], m[2],
        m[4], m[5], m[6],
        m[8], m[9], m[10]
    ]);
}


// Upload current geometry accumulator to GPU as a batch
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

// Add a quad (4 vertices, CCW winding, auto-computed normal)
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

// Add an axis-aligned box (6 faces, outward normals)
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

// Add a cylinder for barrels
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


// Build all scene geometry
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


// Toggle shading mode
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

// Update camera (physics / input)
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

// Draw one geometry batch
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

// Main render loop
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

// Start the game (called from overlay button)
function startGame() {
    document.getElementById('overlay').style.display = 'none';
    gl.canvas.requestPointerLock();
    requestAnimFrame(render);
}


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
