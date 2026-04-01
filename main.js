var gl;
var program;

var attribPos, attribNorm, attribCol;
var uniformMvp, uniformModel, uniformNormalMatrix, uniformMode, uniformLightPos, uniformEye, uniformTime;

var batches = [];
var gunBatch = null;
var geomPositions = [], geomNormals = [], geomColors = [], geomIndices = [], geomBaseIndex = 0;

var colors = {
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

var roomWidth = 30, roomDepth = 40, wallHeight = 6;
var halfWidth = 15, halfDepth = 20;

var cam = {
    x: 0, y: 2.2, z: 17,
    yaw: 0, pitch: 0, roll: 0,
    fov: 75, near: 0.15, far: 250,
    speed: 5
};

var BOUNDS = {
    minX: -halfWidth + 0.4, maxX:  halfWidth - 0.4,
    minZ: -halfDepth + 0.4, maxZ:  halfDepth + 18 - 0.4,
    minY:  1.6, maxY:  wallHeight - 0.4
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
    result[0] = (2 * n) / (r - l);
    result[5] = (2 * n) / (t - b);
    result[8] = (r + l) / (r - l);
    result[9] = (t + b) / (t - b);
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
    if (geomPositions.length === 0) return;

    var posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geomPositions), gl.STATIC_DRAW);

    var normalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geomNormals), gl.STATIC_DRAW);

    var colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geomColors), gl.STATIC_DRAW);

    var indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(geomIndices), gl.STATIC_DRAW);
    var edgeSet = {};
    var wireIndices = [];
    for (var i = 0; i < geomIndices.length; i += 3) {
        var a = geomIndices[i], b = geomIndices[i+1], c = geomIndices[i+2];
        var edges = [[a,b],[b,c],[a,c]];
        for (var j = 0; j < edges.length; j++) {
            var p = edges[j][0], q = edges[j][1];
            var key = Math.min(p,q) + '|' + Math.max(p,q);
            if (!edgeSet[key]) {
                edgeSet[key] = true;
                wireIndices.push(p, q);
            }
        }
    }
    var wireIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, wireIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(wireIndices), gl.STATIC_DRAW);

    batches.push({
        posBuffer: posBuffer, normalBuffer: normalBuffer, colorBuffer: colorBuffer,
        indexBuffer: indexBuffer, indexCount: geomIndices.length,
        wireIndexBuffer: wireIndexBuffer, wireCount: wireIndices.length
    });

    geomPositions = [];  geomNormals = [];  geomColors = [];  geomIndices = [];  geomBaseIndex = 0;
}

function addQuad(p0, p1, p2, p3, col) {
    if (geomBaseIndex + 4 > 62000) flushBatch();

    var e1 = subtract(p1, p0);
    var e2 = subtract(p3, p0);
    var n  = normalize(cross(e1, e2));

    var verts = [p0, p1, p2, p3];
    for (var i = 0; i < 4; i++) {
        geomPositions.push(verts[i][0], verts[i][1], verts[i][2]);
        geomNormals.push(n[0], n[1], n[2]);
        geomColors.push(col[0], col[1], col[2]);
    }
    geomIndices.push(geomBaseIndex, geomBaseIndex+1, geomBaseIndex+2, geomBaseIndex, geomBaseIndex+2, geomBaseIndex+3);
    geomBaseIndex += 4;
}

function addSmoothQuad(p0, p1, p2, p3, n0, n1, n2, n3, col) {
    if (geomBaseIndex + 4 > 62000) flushBatch();
    var verts = [p0, p1, p2, p3];
    var norms = [n0, n1, n2, n3];
    for (var i = 0; i < 4; i++) {
        geomPositions.push(verts[i][0], verts[i][1], verts[i][2]);
        geomNormals.push(norms[i][0], norms[i][1], norms[i][2]);
        geomColors.push(col[0], col[1], col[2]);
    }
    geomIndices.push(geomBaseIndex, geomBaseIndex + 1, geomBaseIndex + 2, geomBaseIndex, geomBaseIndex + 2, geomBaseIndex + 3);
    geomBaseIndex += 4;
}

function addBox(x, y, z, W, H, D, sideCol, topCol, botCol) {
    var sideColor = sideCol;
    var topColor = topCol || sideColor;
    var bottomColor = botCol || sideColor;
    var x1 = x+W, y1 = y+H, z1 = z+D;

    addQuad([x,y,z1],[x,y,z], [x1,y,z],  [x1,y,z1], bottomColor);
    addQuad([x,y1,z],[x,y1,z1],[x1,y1,z1],[x1,y1,z], topColor);
    addQuad([x,y,z1],[x1,y,z1],[x1,y1,z1],[x,y1,z1], sideColor);
    addQuad([x1,y,z],[x,y,z], [x,y1,z],  [x1,y1,z], sideColor);
    addQuad([x,y,z], [x,y,z1], [x,y1,z1], [x,y1,z],  sideColor);
    addQuad([x1,y,z1],[x1,y,z],[x1,y1,z],[x1,y1,z1], sideColor);
}

function addCylinder(x, y, z, radius, height, sides, sideCol, capCol) {
    var topY = y + height;
    var botY = y;
    var step = (Math.PI * 2) / sides;

    for (var i = 0; i < sides; i++) {
        var angle1 = i * step;
        var angle2 = (i + 1) * step;
        var x1 = x + Math.cos(angle1) * radius;
        var z1 = z + Math.sin(angle1) * radius;
        var x2 = x + Math.cos(angle2) * radius;
        var z2 = z + Math.sin(angle2) * radius;
        var normal1 = [Math.cos(angle1), 0, Math.sin(angle1)];
        var normal2 = [Math.cos(angle2), 0, Math.sin(angle2)];

        addSmoothQuad([x2, botY, z2], [x1, botY, z1], [x1, topY, z1], [x2, topY, z2], normal2, normal1, normal1, normal2, sideCol);
        addQuad([x,topY,z],[x2,topY,z2],[x1,topY,z1],[x1,topY,z1], capCol);
        addQuad([x,botY,z],[x1,botY,z1],[x2,botY,z2],[x2,botY,z2], capCol);
    }
}

function buildScene() {
    var trimLow = 0.85, trimHigh = 1.15;
    addQuad([-halfWidth,0, halfDepth],[ halfWidth,0, halfDepth],
            [ halfWidth,0,-halfDepth],[-halfWidth,0,-halfDepth], colors.floor);
    addQuad([-halfWidth,wallHeight,-halfDepth],[ halfWidth,wallHeight,-halfDepth],
            [ halfWidth,wallHeight, halfDepth],[-halfWidth,wallHeight, halfDepth], colors.ceil);
    addQuad([ halfWidth,0,-halfDepth],[-halfWidth,0,-halfDepth],
            [-halfWidth,wallHeight,-halfDepth],[ halfWidth,wallHeight,-halfDepth], colors.w1);
    addQuad([-halfWidth,0, halfDepth],[ halfWidth,0, halfDepth],
            [ halfWidth,wallHeight, halfDepth],[-halfWidth,wallHeight, halfDepth], colors.w2);
    addQuad([-halfWidth,0,-halfDepth],[-halfWidth,0, halfDepth],
            [-halfWidth,wallHeight, halfDepth],[-halfWidth,wallHeight,-halfDepth], colors.w3);
    addQuad([ halfWidth,0, halfDepth],[ halfWidth,0,-halfDepth],
            [ halfWidth,wallHeight,-halfDepth],[ halfWidth,wallHeight, halfDepth], colors.w3);
    addQuad([ halfWidth,trimLow,-halfDepth+0.01],[-halfWidth,trimLow,-halfDepth+0.01],
            [-halfWidth,trimHigh,-halfDepth+0.01],[ halfWidth,trimHigh,-halfDepth+0.01], colors.acc);
    addQuad([-halfWidth,trimLow, halfDepth-0.01],[ halfWidth,trimLow, halfDepth-0.01],
            [ halfWidth,trimHigh, halfDepth-0.01],[-halfWidth,trimHigh, halfDepth-0.01], colors.acc);
    addQuad([-halfWidth+0.01,trimLow,-halfDepth],[-halfWidth+0.01,trimLow, halfDepth],
            [-halfWidth+0.01,trimHigh, halfDepth],[-halfWidth+0.01,trimHigh,-halfDepth], colors.acc);
    addQuad([ halfWidth-0.01,trimLow, halfDepth],[ halfWidth-0.01,trimLow,-halfDepth],
            [ halfWidth-0.01,trimHigh,-halfDepth],[ halfWidth-0.01,trimHigh, halfDepth], colors.acc);
    for (var beamZ = -16; beamZ <= 16; beamZ += 8) {
        addBox(-halfWidth, wallHeight-0.32, beamZ-0.2, roomWidth, 0.31, 0.4, colors.dark);
    }
    for (var trimIdx = -2; trimIdx <= 2; trimIdx++) {
        addBox(trimIdx*7 - 0.05, 0.005, -halfDepth, 0.1, 0.07, roomDepth, colors.trim);
    }
    var columnPositions = [
        [-9, -14], [ 9, -14],
        [-9, -4], [ 9, -4],
        [-9, 6], [ 9, 6],
        [-5, 15], [ 5, 15]
    ];
    for (var colIdx = 0; colIdx < columnPositions.length; colIdx++) {
        var colX = columnPositions[colIdx][0], colZ = columnPositions[colIdx][1];
        addBox(colX-0.9,  0, colZ-0.9,  1.8, 0.28, 1.8, colors.cap, colors.cap, colors.cap);
        addBox(colX-0.55, 0.28, colZ-0.55, 1.1, wallHeight-0.56, 1.1, colors.col);
        addBox(colX-0.9,  wallHeight-0.28, colZ-0.9, 1.8, 0.28, 1.8, colors.cap, colors.cap, colors.cap);
    }
    var lavaX = 0, lavaZ = 5;
    addBox(lavaX-5.5, 0.01, lavaZ-5.5, 11, 0.25, 11, colors.w3, colors.step, colors.w3);
    addBox(lavaX-3.5, 0.01, lavaZ-3.5,  7, 0.50,  7, colors.step, colors.acc, colors.step);
    addQuad([lavaX-2.4,0.52,lavaZ+2.4],[lavaX+2.4,0.52,lavaZ+2.4],
            [lavaX+2.4,0.52,lavaZ-2.4],[lavaX-2.4,0.52,lavaZ-2.4], colors.lava);
    addQuad([lavaX-2.4,0.50,lavaZ-2.4],[lavaX-2.4,0.50,lavaZ+2.4],
            [lavaX-2.4,0.52,lavaZ+2.4],[lavaX-2.4,0.52,lavaZ-2.4], colors.lava);
    addQuad([lavaX+2.4,0.50,lavaZ+2.4],[lavaX+2.4,0.50,lavaZ-2.4],
            [lavaX+2.4,0.52,lavaZ-2.4],[lavaX+2.4,0.52,lavaZ+2.4], colors.lava);
    addQuad([lavaX-2.4,0.50,lavaZ+2.4],[lavaX+2.4,0.50,lavaZ+2.4],
            [lavaX+2.4,0.52,lavaZ+2.4],[lavaX-2.4,0.52,lavaZ+2.4], colors.acc);
    addQuad([lavaX+2.4,0.50,lavaZ-2.4],[lavaX-2.4,0.50,lavaZ-2.4],
            [lavaX-2.4,0.52,lavaZ-2.4],[lavaX+2.4,0.52,lavaZ-2.4], colors.acc);
    addBox(-halfWidth+0.01, 0.005, -16, 5, 0.45, 7, colors.plat, colors.floor, colors.w2);
    addBox( halfWidth-5.01, 0.005, -16, 5, 0.45, 7, colors.plat, colors.floor, colors.w2);
    var alcoveX = 3.5, alcoveZ = 1.5;
    addQuad([-alcoveX,0,-halfDepth-alcoveZ],[ alcoveX,0,-halfDepth-alcoveZ],
            [ alcoveX,0,-halfDepth],[-alcoveX,0,-halfDepth], colors.floor);
    addQuad([-alcoveX,wallHeight,-halfDepth],[ alcoveX,wallHeight,-halfDepth],
            [ alcoveX,wallHeight,-halfDepth-alcoveZ],[-alcoveX,wallHeight,-halfDepth-alcoveZ], colors.ceil);
    addQuad([ alcoveX,0,-halfDepth-alcoveZ],[-alcoveX,0,-halfDepth-alcoveZ],
            [-alcoveX,wallHeight,-halfDepth-alcoveZ],[ alcoveX,wallHeight,-halfDepth-alcoveZ], colors.w1);
    addQuad([-alcoveX,0,-halfDepth],[-alcoveX,0,-halfDepth-alcoveZ],
            [-alcoveX,wallHeight,-halfDepth-alcoveZ],[-alcoveX,wallHeight,-halfDepth], colors.w4);
    addQuad([ alcoveX,0,-halfDepth-alcoveZ],[ alcoveX,0,-halfDepth],
            [ alcoveX,wallHeight,-halfDepth],[ alcoveX,wallHeight,-halfDepth-alcoveZ], colors.w4);
    addQuad([ halfWidth,0,-halfDepth],[ alcoveX,0,-halfDepth],
            [ alcoveX,wallHeight,-halfDepth],[ halfWidth,wallHeight,-halfDepth], colors.w1);
    addQuad([-alcoveX,0,-halfDepth],[-halfWidth,0,-halfDepth],
            [-halfWidth,wallHeight,-halfDepth],[-alcoveX,wallHeight,-halfDepth], colors.w1);
    var hallWidth = 7, hallDepth = 18;
    var halfHall = hallWidth / 2;

    addQuad([-halfHall,0,halfDepth+hallDepth],[ halfHall,0,halfDepth+hallDepth],
            [ halfHall,0,halfDepth],[-halfHall,0,halfDepth], colors.floor);
    addQuad([-halfHall,wallHeight,halfDepth],[ halfHall,wallHeight,halfDepth],
            [ halfHall,wallHeight,halfDepth+hallDepth],[-halfHall,wallHeight,halfDepth+hallDepth], colors.ceil);
    addQuad([-halfHall,0,halfDepth],[-halfHall,0,halfDepth+hallDepth],
            [-halfHall,wallHeight,halfDepth+hallDepth],[-halfHall,wallHeight,halfDepth], colors.w4);
    addQuad([ halfHall,0,halfDepth+hallDepth],[ halfHall,0,halfDepth],
            [ halfHall,wallHeight,halfDepth],[ halfHall,wallHeight,halfDepth+hallDepth], colors.w4);
    addQuad([ halfHall,0,halfDepth+hallDepth],[-halfHall,0,halfDepth+hallDepth],
            [-halfHall,wallHeight,halfDepth+hallDepth],[ halfHall,wallHeight,halfDepth+hallDepth], colors.w1);
    addQuad([-halfHall+0.01,trimLow,halfDepth],[-halfHall+0.01,trimLow,halfDepth+hallDepth],
            [-halfHall+0.01,trimHigh,halfDepth+hallDepth],[-halfHall+0.01,trimHigh,halfDepth], colors.acc);
    addQuad([ halfHall-0.01,trimLow,halfDepth+hallDepth],[ halfHall-0.01,trimLow,halfDepth],
            [ halfHall-0.01,trimHigh,halfDepth],[ halfHall-0.01,trimHigh,halfDepth+hallDepth], colors.acc);
    addQuad([-halfWidth,0,halfDepth],[-halfHall,0,halfDepth],
            [-halfHall,wallHeight,halfDepth],[-halfWidth,wallHeight,halfDepth], colors.w2);
    addQuad([ halfHall,0,halfDepth],[ halfWidth,0,halfDepth],
            [ halfWidth,wallHeight,halfDepth],[ halfHall,wallHeight,halfDepth], colors.w2);

    flushBatch();
    var pillarPositions = [
        [-9, -14], [9, -14],
        [-9, -4], [9, -4],
        [-9, 6], [9, 6],
        [-5, 15], [5, 15]
    ];
    var barrelCount = 3;
    var placed = 0;
    var attempts = 0;
    while (placed < barrelCount && attempts < 200) {
        var barrelX = (Math.random() - 0.5) * (roomWidth - 6);
        var barrelZ = (Math.random() - 0.5) * (roomDepth - 10);
        var deltaX = barrelX - 0, deltaZ = barrelZ - 5;
        var distToLava = Math.sqrt(deltaX * deltaX + deltaZ * deltaZ);
        var hitPillar = false;
        for (var pillarIdx = 0; pillarIdx < pillarPositions.length; pillarIdx++) {
            var pillarX = pillarPositions[pillarIdx][0], pillarZ = pillarPositions[pillarIdx][1];
            if (Math.abs(barrelX - pillarX) < 1.8 && Math.abs(barrelZ - pillarZ) < 1.8) {
                hitPillar = true;
                break;
            }
        }
        if (distToLava >= 6 && !hitPillar) {
            addCylinder(barrelX, 0, barrelZ, 0.6, 1.6, 10, colors.barrel, colors.lid);
            placed++;
        }
        attempts++;
    }

    flushBatch();
}
function buildGun() {
    geomPositions = []; geomNormals = []; geomColors = []; geomIndices = []; geomBaseIndex = 0;

    var metal  = [0.15, 0.15, 0.17];
    var metal2 = [0.28, 0.28, 0.33];
    var wood   = [0.42, 0.24, 0.08];
    var wood2  = [0.35, 0.18, 0.06];
    var black  = [0.05, 0.05, 0.05];
    var steel  = [0.55, 0.55, 0.60];
    addBox(-0.04, 0.02, -0.80, 0.08, 0.07, 0.82, metal, metal2, metal);
    addBox(-0.012, 0.09, -0.78, 0.024, 0.018, 0.76, steel, steel, steel);
    addBox(-0.014, 0.108, -0.77, 0.028, 0.020, 0.028, steel, steel, steel);
    addBox(-0.052, -0.01, -0.60, 0.104, 0.055, 0.24, wood2, wood2, wood2);
    addBox(-0.060, -0.04, -0.04, 0.12, 0.14, 0.30, metal, metal2, metal);
    addBox( 0.058, -0.01, -0.03, 0.008, 0.07, 0.16, black, black, black);
    addBox(-0.028, -0.10,  0.00, 0.056, 0.010, 0.14, black, black, black);
    addBox(-0.028, -0.18,  0.00, 0.056, 0.080, 0.01, black, black, black);
    addBox(-0.028, -0.18,  0.13, 0.056, 0.080, 0.01, black, black, black);
    addBox(-0.038, -0.42,  0.02, 0.076, 0.30, 0.09, wood, wood, wood);
    addBox(-0.048,  0.00,  0.26, 0.096, 0.09, 0.28, wood, wood, wood);
    addBox(-0.052, -0.01,  0.54, 0.104, 0.10, 0.018, metal, metal, metal);
    addBox(-0.042,  0.018, -0.82, 0.084, 0.074, 0.018, black, black, black);

    flushBatch();
    gunBatch = batches.pop();
}

function setShadingMode(mode) {
    shadingMode = mode;
    var btns = document.querySelectorAll('.sbtn');
    for (var i = 0; i < btns.length; i++) {
        if (i === mode) btns[i].className = 'sbtn on';
        else btns[i].className = 'sbtn';
    }
    document.getElementById('hshd').textContent =
        'SHADING: ' + ['WIREFRAME', 'FLAT', 'SMOOTH'][mode];
}

function update(dt) {
    var moveSpeed = cam.speed * ((keys.ShiftLeft || keys.ShiftRight) ? 2.5 : 1.0);

    var cosYaw = Math.cos(cam.yaw), sinYaw = Math.sin(cam.yaw);
    var forwardX = sinYaw,  forwardZ = -cosYaw;
    var rightX = cosYaw,  rightZ = sinYaw;

    if (keys.KeyW) { cam.x += forwardX*moveSpeed*dt;  cam.z += forwardZ*moveSpeed*dt; }
    if (keys.KeyS) { cam.x -= forwardX*moveSpeed*dt;  cam.z -= forwardZ*moveSpeed*dt; }
    if (keys.KeyA) { cam.x -= rightX*moveSpeed*dt;  cam.z -= rightZ*moveSpeed*dt; }
    if (keys.KeyD) { cam.x += rightX*moveSpeed*dt;  cam.z += rightZ*moveSpeed*dt; }
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
    gl.bindBuffer(gl.ARRAY_BUFFER, b.posBuffer);
    gl.enableVertexAttribArray(attribPos);
    gl.vertexAttribPointer(attribPos, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, b.normalBuffer);
    gl.enableVertexAttribArray(attribNorm);
    gl.vertexAttribPointer(attribNorm, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, b.colorBuffer);
    gl.enableVertexAttribArray(attribCol);
    gl.vertexAttribPointer(attribCol, 3, gl.FLOAT, false, 0, 0);

    if (shadingMode === 0) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.wireIndexBuffer);
        gl.drawElements(gl.LINES, b.wireCount, gl.UNSIGNED_SHORT, 0);
    } else {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.indexBuffer);
        gl.drawElements(gl.TRIANGLES, b.indexCount, gl.UNSIGNED_SHORT, 0);
    }
}

function render(now) {
    requestAnimFrame(render);

    var dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    fpsAcc += dt;  fpsCnt++;
    if (fpsAcc >= 0.5) { fps = Math.round(fpsCnt / fpsAcc); fpsAcc = 0; fpsCnt = 0; }

    update(dt);
    var radToDeg = 180.0 / Math.PI;
    var translateMat = translate(-cam.x, -cam.y, -cam.z);
    var rotYawMat = rotateY(cam.yaw * radToDeg);
    var rotPitchMat = rotateX(cam.pitch * radToDeg);
    var rotRollMat = rotateZ(cam.roll * radToDeg);
    var view = mult(rotRollMat, mult(rotPitchMat, mult(rotYawMat, translateMat)));
    var aspectRatio = gl.canvas.width / gl.canvas.height;
    var fovRadians = cam.fov * Math.PI / 180;
    var halfTanFov = cam.near * Math.tan(fovRadians / 2);
    var halfFrustumW = halfTanFov * aspectRatio;
    var proj = frustum(-halfFrustumW, halfFrustumW, -halfTanFov, halfTanFov, cam.near, cam.far);

    var mvpMatrix = mult(proj, view);
    var modelMatrix = identity();
    var normalMatrix  = mat3Normal(modelMatrix);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.uniformMatrix4fv(uniformMvp, false, new Float32Array(mvpMatrix));
    gl.uniformMatrix4fv(uniformModel, false, new Float32Array(modelMatrix));
    gl.uniformMatrix3fv(uniformNormalMatrix, false, normalMatrix);
    gl.uniform1i(uniformMode, shadingMode);
    gl.uniform3f(uniformLightPos, 0, 4.5, 0);
    gl.uniform3f(uniformEye, cam.x, cam.y, cam.z);
    gl.uniform1f(uniformTime, now * 0.001);

    for (var i = 0; i < batches.length; i++) {
        drawBatch(batches[i]);
    }

    if (gunBatch) {
        gl.disable(gl.DEPTH_TEST);
        gl.clear(gl.DEPTH_BUFFER_BIT);

        var gunModelMatrix = identity();
        gunModelMatrix = mult(gunModelMatrix, translate(0.35, -0.50, -0.70));
        gunModelMatrix = mult(gunModelMatrix, rotateZ(0));
        gunModelMatrix = mult(gunModelMatrix, rotateX(0));
        gunModelMatrix = mult(gunModelMatrix, rotateY(4));
        gunModelMatrix = mult(gunModelMatrix, scalem(1.10, 1.10, 1.10));
        
        var gunProjection = frustum(-0.13 * aspectRatio, 0.13 * aspectRatio, -0.13, 0.13, 0.13, 10.0);
        var gunMvpMatrix  = mult(gunProjection, gunModelMatrix);
        var gunNormalMatrix   = mat3Normal(gunModelMatrix);

        gl.uniformMatrix4fv(uniformMvp, false, new Float32Array(gunMvpMatrix));
        gl.uniformMatrix4fv(uniformModel, false, new Float32Array(gunModelMatrix));
        gl.uniformMatrix3fv(uniformNormalMatrix, false, gunNormalMatrix);
        gl.uniform1i(uniformMode, shadingMode === 0 ? 0 : 2);
        gl.uniform3f(uniformLightPos, 0.5, 2.0, 1.0);
        gl.uniform3f(uniformEye, 0, 0, 0);

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

    attribPos = gl.getAttribLocation(program, 'aPos');
    attribNorm = gl.getAttribLocation(program, 'aNorm');
    attribCol = gl.getAttribLocation(program, 'aCol');

    uniformMvp = gl.getUniformLocation(program, 'uMVP');
    uniformModel = gl.getUniformLocation(program, 'uModel');
    uniformNormalMatrix = gl.getUniformLocation(program, 'uNM');
    uniformMode = gl.getUniformLocation(program, 'uMode');
    uniformLightPos = gl.getUniformLocation(program, 'uLightPos');
    uniformEye = gl.getUniformLocation(program, 'uEye');
    uniformTime = gl.getUniformLocation(program, 'uTime');

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
        if (e.code === 'Digit1') setShadingMode(0);
        if (e.code === 'Digit2') setShadingMode(1);
        if (e.code === 'Digit3') setShadingMode(2);
    });
};