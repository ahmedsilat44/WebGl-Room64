var gl;
var program;

var aPos, aNorm, aCol;
var uMVP, uModel, uNM, uMode, uLightPos, uEye, uTime;

var batches = [];
var gunBatch = null;
var gP = [], gN = [], gC = [], gI = [], gBase = 0;

var C = {
    floor:  [0.35, 0.33, 0.30],
    ceil:   [0.42, 0.40, 0.37],
    w1:     [0.45, 0.42, 0.38],
    w2:     [0.40, 0.38, 0.35],
    w3:     [0.52, 0.43, 0.30],
    w4:     [0.48, 0.38, 0.26],
    col:    [0.48, 0.50, 0.56],
    cap:    [0.52, 0.54, 0.60],
    acc:    [0.55, 0.08, 0.04],
    lava:   [0.72, 0.28, 0.01],
    dark:   [0.35, 0.22, 0.10],
    trim:   [0.45, 0.30, 0.14],
    step:   [0.22, 0.12, 0.08],
    plat:   [0.44, 0.42, 0.38],
    barrel: [0.50, 0.18, 0.10],
    lid:    [0.38, 0.14, 0.08]
};

var RoomWidth = 30, RoomDepth = 40, WallHeight = 6;
var halfWidth = 15, halfDepth = 20;

var cam = {
    x: 0,   y: 2.2,  z: 17,
    yaw: 0, pitch: 0, roll: 0,
    fov: 75, near: 0.15, far: 250,
    speed: 5
};

var BOUNDS = {
    minX: -halfWidth + 0.4,  maxX:  halfWidth - 0.4,
    minZ: -halfDepth + 0.4,  maxZ:  halfDepth + 18 - 0.4,
    minY:  1.6,              maxY:  WallHeight - 0.4
};
var HALL = {
    halfWidth: 3.5,
    startZ: halfDepth,
    pad: 0.4
};

var keys = {};
var pointerLocked = false;
var shadingMode = 1;
var lastTime = 0, fps = 0, fpsAcc = 0, fpsCnt = 0;

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

function mat3Normal(m) {
    return new Float32Array([
        m[0], m[1], m[2],
        m[4], m[5], m[6],
        m[8], m[9], m[10]
    ]);
}

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

function addSmoothQuad(p0, p1, p2, p3, n0, n1, n2, n3, col) {
    if (gBase + 4 > 62000) flushBatch();
    var verts = [p0, p1, p2, p3];
    var norms = [n0, n1, n2, n3];
    for (var i = 0; i < 4; i++) {
        gP.push(verts[i][0], verts[i][1], verts[i][2]);
        gN.push(norms[i][0], norms[i][1], norms[i][2]);
        gC.push(col[0], col[1], col[2]);
    }
    gI.push(gBase, gBase + 1, gBase + 2, gBase, gBase + 2, gBase + 3);
    gBase += 4;
}

function addBox(x, y, z, W, H, D, sideCol, topCol, botCol) {
    var sc = sideCol;
    var tc = topCol || sc;
    var bc = botCol || sc;
    var x1 = x+W, y1 = y+H, z1 = z+D;

    addQuad([x,y,z1],[x,y,z],  [x1,y,z],  [x1,y,z1], bc);
    addQuad([x,y1,z],[x,y1,z1],[x1,y1,z1],[x1,y1,z], tc);
    addQuad([x,y,z1],[x1,y,z1],[x1,y1,z1],[x,y1,z1], sc);
    addQuad([x1,y,z],[x,y,z],  [x,y1,z],  [x1,y1,z], sc);
    addQuad([x,y,z], [x,y,z1], [x,y1,z1], [x,y1,z],  sc);
    addQuad([x1,y,z1],[x1,y,z],[x1,y1,z],[x1,y1,z1], sc);
}

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
        var n1 = [Math.cos(a1), 0, Math.sin(a1)];
        var n2 = [Math.cos(a2), 0, Math.sin(a2)];

        addSmoothQuad([x2, botY, z2], [x1, botY, z1], [x1, topY, z1], [x2, topY, z2], n2, n1, n1, n2, sideCol);
        addQuad([x,topY,z],[x2,topY,z2],[x1,topY,z1],[x1,topY,z1], capCol);
        addQuad([x,botY,z],[x1,botY,z1],[x2,botY,z2],[x2,botY,z2], capCol);
    }
}

function buildScene() {
    var T0 = 0.85, T1 = 1.15;
    addQuad([-halfWidth,0, halfDepth],[ halfWidth,0, halfDepth],
            [ halfWidth,0,-halfDepth],[-halfWidth,0,-halfDepth], C.floor);
    addQuad([-halfWidth,WallHeight,-halfDepth],[ halfWidth,WallHeight,-halfDepth],
            [ halfWidth,WallHeight, halfDepth],[-halfWidth,WallHeight, halfDepth], C.ceil);
    addQuad([ halfWidth,0,-halfDepth],[-halfWidth,0,-halfDepth],
            [-halfWidth,WallHeight,-halfDepth],[ halfWidth,WallHeight,-halfDepth], C.w1);
    addQuad([-halfWidth,0, halfDepth],[ halfWidth,0, halfDepth],
            [ halfWidth,WallHeight, halfDepth],[-halfWidth,WallHeight, halfDepth], C.w2);
    addQuad([-halfWidth,0,-halfDepth],[-halfWidth,0, halfDepth],
            [-halfWidth,WallHeight, halfDepth],[-halfWidth,WallHeight,-halfDepth], C.w3);
    addQuad([ halfWidth,0, halfDepth],[ halfWidth,0,-halfDepth],
            [ halfWidth,WallHeight,-halfDepth],[ halfWidth,WallHeight, halfDepth], C.w3);
    addQuad([ halfWidth,T0,-halfDepth+0.01],[-halfWidth,T0,-halfDepth+0.01],
            [-halfWidth,T1,-halfDepth+0.01],[ halfWidth,T1,-halfDepth+0.01], C.acc);
    addQuad([-halfWidth,T0, halfDepth-0.01],[ halfWidth,T0, halfDepth-0.01],
            [ halfWidth,T1, halfDepth-0.01],[-halfWidth,T1, halfDepth-0.01], C.acc);
    addQuad([-halfWidth+0.01,T0,-halfDepth],[-halfWidth+0.01,T0, halfDepth],
            [-halfWidth+0.01,T1, halfDepth],[-halfWidth+0.01,T1,-halfDepth], C.acc);
    addQuad([ halfWidth-0.01,T0, halfDepth],[ halfWidth-0.01,T0,-halfDepth],
            [ halfWidth-0.01,T1,-halfDepth],[ halfWidth-0.01,T1, halfDepth], C.acc);
    for (var bz = -16; bz <= 16; bz += 8) {
        addBox(-halfWidth, WallHeight-0.32, bz-0.2, RoomWidth, 0.31, 0.4, C.dark);
    }
    for (var i = -2; i <= 2; i++) {
        addBox(i*7 - 0.05, 0.005, -halfDepth, 0.1, 0.07, RoomDepth, C.trim);
    }
    var columnPositions = [
        [-9, -14], [ 9, -14],
        [-9,  -4], [ 9,  -4],
        [-9,   6], [ 9,   6],
        [-5,  15], [ 5,  15]
    ];
    for (var ci = 0; ci < columnPositions.length; ci++) {
        var cx = columnPositions[ci][0], cz = columnPositions[ci][1];
        addBox(cx-0.9,  0,              cz-0.9,  1.8, 0.28, 1.8, C.cap, C.cap, C.cap);
        addBox(cx-0.55, 0.28,           cz-0.55, 1.1, WallHeight-0.56, 1.1, C.col);
        addBox(cx-0.9,  WallHeight-0.28, cz-0.9, 1.8, 0.28, 1.8, C.cap, C.cap, C.cap);
    }
    var LX = 0, LZ = 5;
    addBox(LX-5.5, 0.01, LZ-5.5, 11, 0.25, 11, C.w3, C.step, C.w3);
    addBox(LX-3.5, 0.01, LZ-3.5,  7, 0.50,  7, C.step, C.acc, C.step);
    addQuad([LX-2.4,0.52,LZ+2.4],[LX+2.4,0.52,LZ+2.4],
            [LX+2.4,0.52,LZ-2.4],[LX-2.4,0.52,LZ-2.4], C.lava);
    addQuad([LX-2.4,0.50,LZ-2.4],[LX-2.4,0.50,LZ+2.4],
            [LX-2.4,0.52,LZ+2.4],[LX-2.4,0.52,LZ-2.4], C.lava);
    addQuad([LX+2.4,0.50,LZ+2.4],[LX+2.4,0.50,LZ-2.4],
            [LX+2.4,0.52,LZ-2.4],[LX+2.4,0.52,LZ+2.4], C.lava);
    addQuad([LX-2.4,0.50,LZ+2.4],[LX+2.4,0.50,LZ+2.4],
            [LX+2.4,0.52,LZ+2.4],[LX-2.4,0.52,LZ+2.4], C.acc);
    addQuad([LX+2.4,0.50,LZ-2.4],[LX-2.4,0.50,LZ-2.4],
            [LX-2.4,0.52,LZ-2.4],[LX+2.4,0.52,LZ-2.4], C.acc);
    addBox(-halfWidth+0.01, 0.005, -16, 5, 0.45, 7, C.plat, C.floor, C.w2);
    addBox( halfWidth-5.01, 0.005, -16, 5, 0.45, 7, C.plat, C.floor, C.w2);
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
    addQuad([-halfWidth,0,halfDepth],[-halfHall,0,halfDepth],
            [-halfHall,WallHeight,halfDepth],[-halfWidth,WallHeight,halfDepth], C.w2);
    addQuad([ halfHall,0,halfDepth],[ halfWidth,0,halfDepth],
            [ halfWidth,WallHeight,halfDepth],[ halfHall,WallHeight,halfDepth], C.w2);

    flushBatch();
    var columnPositions2 = [
        [-9, -14], [ 9, -14],
        [-9,  -4], [ 9,  -4],
        [-9,   6], [ 9,   6],
        [-5,  15], [ 5,  15]
    ];
    var barrelCount = 3;
    var placed = 0;
    var attempts = 0;
    while (placed < barrelCount && attempts < 200) {
        var bx = (Math.random() - 0.5) * (RoomWidth - 6);
        var bz = (Math.random() - 0.5) * (RoomDepth - 10);
        var dx = bx - 0, dz = bz - 5;
        var distToLava = Math.sqrt(dx * dx + dz * dz);
        var hitPillar = false;
        for (var pi = 0; pi < columnPositions2.length; pi++) {
            var px = columnPositions2[pi][0], pz = columnPositions2[pi][1];
            if (Math.abs(bx - px) < 1.8 && Math.abs(bz - pz) < 1.8) {
                hitPillar = true;
                break;
            }
        }
        if (distToLava >= 6 && !hitPillar) {
            addCylinder(bx, 0, bz, 0.6, 1.6, 10, C.barrel, C.lid);
            placed++;
        }
        attempts++;
    }

    flushBatch();
}
function buildGun() {
    gP = []; gN = []; gC = []; gI = []; gBase = 0;

    var metal  = [0.15, 0.15, 0.17];
    var metal2 = [0.28, 0.28, 0.33];
    var wood   = [0.42, 0.24, 0.08];
    var wood2  = [0.35, 0.18, 0.06];
    var black  = [0.05, 0.05, 0.05];
    var steel  = [0.55, 0.55, 0.60];
    addBox(-0.04, 0.02, -0.80,  0.08, 0.07, 0.82, metal, metal2, metal);
    addBox(-0.012, 0.09, -0.78,  0.024, 0.018, 0.76, steel, steel, steel);
    addBox(-0.014, 0.108, -0.77,  0.028, 0.020, 0.028, steel, steel, steel);
    addBox(-0.052, -0.01, -0.60,  0.104, 0.055, 0.24, wood2, wood2, wood2);
    addBox(-0.060, -0.04, -0.04,  0.12, 0.14, 0.30, metal, metal2, metal);
    addBox( 0.058, -0.01, -0.03,  0.008, 0.07, 0.16, black, black, black);
    addBox(-0.028, -0.10,  0.00,  0.056, 0.010, 0.14, black, black, black);
    addBox(-0.028, -0.18,  0.00,  0.056, 0.080, 0.01, black, black, black);
    addBox(-0.028, -0.18,  0.13,  0.056, 0.080, 0.01, black, black, black);
    addBox(-0.038, -0.42,  0.02,  0.076, 0.30, 0.09, wood, wood, wood);
    addBox(-0.048,  0.00,  0.26,  0.096, 0.09, 0.28, wood, wood, wood);
    addBox(-0.052, -0.01,  0.54,  0.104, 0.10, 0.018, metal, metal, metal);
    addBox(-0.042,  0.018, -0.82,  0.084, 0.074, 0.018, black, black, black);

    flushBatch();
    gunBatch = batches.pop();
}

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

    if (keys.Comma)  cam.speed = Math.max(1,  cam.speed - 3*dt);
    if (keys.Period) cam.speed = Math.min(25, cam.speed + 3*dt);

    cam.z = Math.max(BOUNDS.minZ, Math.min(BOUNDS.maxZ, cam.z));
    var hallMinX = -HALL.halfWidth + HALL.pad;
    var hallMaxX =  HALL.halfWidth - HALL.pad;
    if (cam.z > HALL.startZ - HALL.pad) {
        cam.x = Math.max(hallMinX, Math.min(hallMaxX, cam.x));
    } else {
        cam.x = Math.max(BOUNDS.minX, Math.min(BOUNDS.maxX, cam.x));
    }
    cam.y = Math.max(BOUNDS.minY, Math.min(BOUNDS.maxY, cam.y));
}

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

function render(now) {
    requestAnimFrame(render);

    var dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    fpsAcc += dt;  fpsCnt++;
    if (fpsAcc >= 0.5) { fps = Math.round(fpsCnt / fpsAcc); fpsAcc = 0; fpsCnt = 0; }

    update(dt);
    var RAD = 180.0 / Math.PI;
    var T   = translate(-cam.x, -cam.y, -cam.z);
    var Ry  = rotateY(cam.yaw * RAD);
    var Rx  = rotateX(cam.pitch * RAD);
    var Rz  = rotateZ(cam.roll * RAD);
    var view = mult(Rz, mult(Rx, mult(Ry, T)));
    var asp   = gl.canvas.width / gl.canvas.height;
    var fovR  = cam.fov * Math.PI / 180;
    var tHalf = cam.near * Math.tan(fovR / 2);
    var wHalf = tHalf * asp;
    var proj  = frustum(-wHalf, wHalf, -tHalf, tHalf, cam.near, cam.far);

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
    gl.uniform1f(uTime, now * 0.001);

    for (var i = 0; i < batches.length; i++) {
        drawBatch(batches[i]);
    }

    if (gunBatch) {
        gl.disable(gl.DEPTH_TEST);
        gl.clear(gl.DEPTH_BUFFER_BIT);

        var gunM = identity();
        gunM = mult(gunM, translate(0.35, -0.50, -0.70));
        gunM = mult(gunM, rotateZ(0));
        gunM = mult(gunM, rotateX(0));
        gunM = mult(gunM, rotateY(4));
        gunM = mult(gunM, scalem(1.10, 1.10, 1.10));
        
        var gunProj = frustum(-0.13 * asp, 0.13 * asp, -0.13, 0.13, 0.13, 10.0);
        var gunMVP  = mult(gunProj, gunM);
        var gunNM   = mat3Normal(gunM);

        gl.uniformMatrix4fv(uMVP,   false, new Float32Array(gunMVP));
        gl.uniformMatrix4fv(uModel, false, new Float32Array(gunM));
        gl.uniformMatrix3fv(uNM,    false, gunNM);
        gl.uniform1i(uMode, shadingMode === 0 ? 0 : 2);
        gl.uniform3f(uLightPos, 0.5, 2.0, 1.0);
        gl.uniform3f(uEye, 0, 0, 0);

        drawBatch(gunBatch);

        gl.enable(gl.DEPTH_TEST);
    }

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

function startGame() {
    document.getElementById('overlay').style.display = 'none';
    gl.canvas.requestPointerLock();
    requestAnimFrame(render);
}

window.onload = function init() {
    var canvas = document.getElementById("gl-canvas");
    gl = WebGLUtils.setupWebGL(canvas);

    if (!gl) { alert("WebGL isn't available"); return; }
    gl.getExtension('OES_standard_derivatives');

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

    aPos  = gl.getAttribLocation(program, 'aPos');
    aNorm = gl.getAttribLocation(program, 'aNorm');
    aCol  = gl.getAttribLocation(program, 'aCol');

    uMVP      = gl.getUniformLocation(program, 'uMVP');
    uModel    = gl.getUniformLocation(program, 'uModel');
    uNM       = gl.getUniformLocation(program, 'uNM');
    uMode     = gl.getUniformLocation(program, 'uMode');
    uLightPos = gl.getUniformLocation(program, 'uLightPos');
    uEye      = gl.getUniformLocation(program, 'uEye');
    uTime     = gl.getUniformLocation(program, 'uTime');

    buildScene();
    buildGun();

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
    document.addEventListener('keydown', function(e) {
        if (e.code === 'Digit1') setSh(0);
        if (e.code === 'Digit2') setSh(1);
        if (e.code === 'Digit3') setSh(2);
    });
};