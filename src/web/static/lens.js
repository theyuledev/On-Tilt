// the 3D view's lens: where a point of the plate lands on the canvas for a given camera
const EYE_PX = 1300;  // eye distance in scene px
const PLATE_PX = 270;  // the plate's half width in scene px, whatever it measures in mm
const SCENE_PX = 900;  // scene px across the canvas
const HORIZON = 0.54;  // the point the camera looks at sits this far down the canvas
const TILT_SHOW = 3;  // the real plate barely tilts, so the picture exaggerates it

// p(x, y, z) -> [canvas x, canvas y, depth, px per scene px]; x, y plate mm, z scene px up
function projector(cam, tilt, half, px) {
  const u = PLATE_PX / half, f = px / SCENE_PX, rad = Math.PI / 180;
  const cy = Math.cos(cam.yaw * rad), sy = Math.sin(cam.yaw * rad);
  const cp = Math.cos(cam.pitch * rad), sp = Math.sin(cam.pitch * rad);
  // a positive tilt lowers that axis's + edge
  const lx = Math.sin(TILT_SHOW * tilt[0] * rad), ly = Math.sin(TILT_SHOW * tilt[1] * rad);
  return (x, y, z = 0) => {
    const h = z - (x * lx + y * ly) * u, dx = (x - cam.cx) * u, dy = (y - cam.cy) * u;
    const X = dx * cy - dy * sy, Y = dx * sy + dy * cy;
    const deep = (Y * sp - h * cp) * cam.zoom;
    const k = (f * cam.zoom * EYE_PX) / Math.max(EYE_PX + deep, 50);
    return [px / 2 + X * k, HORIZON * px - (Y * cp + h * sp) * k, deep, k];
  };
}
