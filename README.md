# Project 1: Virtual Worlds - DOOM64 Arena

## How to Run the Project
1. Extract the contents of the submitted `.zip` file into a single folder.
2. Ensure all files (`index.html`, `main.js`, `common.js`, `initShaders.js`, `webgl-utils.js`) are in the same directory.
3. Open `index.html` in any modern web browser (Google Chrome or Mozilla Firefox are recommended).
4. Click the **"[ CLICK TO ENTER ]"** button on the screen. This will lock your mouse to the canvas for a seamless first-person experience.
5. To exit the pointer lock and get your mouse back at any time, simply press the **`ESC`** key.

## Controls

### Movement
* **W / A / S / D:** Move forward, left, backward, and right (relative to your viewing angle).
* **R / F:** Fly vertically up and down.
* **Shift (Left or Right):** Hold to sprint (2.5x speed).
* **`,` (Comma) / `.` (Period):** Decrease or increase the base movement speed.

### Viewing & Camera Orientation
* **Mouse Movement:** Look around (controls Pitch and Yaw).
* **Left / Right Arrows:** Look left/right (Yaw alternative).
* **Q / E:** Roll the camera (bank left/right).

### Dynamic View Bounds
* **Up / Down Arrows:** Zoom in/out by adjusting the Field of View (FOV). This dynamically alters the **Top and Bottom** viewing bounds.
* **`[` / `]` (Brackets):** Adjust the **Near and Far** clipping planes.

### Shading Modes
* **1:** Wireframe Mode (shows structural edges).
* **2:** Flat Shading Mode (faceted look using reconstructed face normals).
* **3:** Smooth Shading Mode (interpolated normals with specular highlights and point lighting).