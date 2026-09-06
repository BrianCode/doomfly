/** Export the user-selected side-view artwork as site and browser logos.
 * Export from the original PNG, using a viewBox to remove empty margins.
 * Run from the site checkout: node design/fly-mark.mjs
 */
import { readFile, writeFile } from 'node:fs/promises';
import sharp from 'sharp';

const publicDir = new URL('../public/', import.meta.url);
const source = await readFile(new URL('pixel-fly.png', publicDir));
const svg = (square = false) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="260 ${square ? -20 : 115} 1040 ${square ? 1040 : 760}"><rect x="260" y="-20" width="1040" height="1040" fill="#000"/><image width="1536" height="1024" href="data:image/png;base64,${source.toString('base64')}" style="image-rendering:pixelated"/></svg>\n`;
await sharp(Buffer.from(svg())).resize(384, 280).png().toFile(new URL('fly-logo.png', publicDir).pathname);
for (const [name, size] of [['fly-icon-96.png', 96], ['apple-touch-icon.png', 192]]) {
  await sharp(Buffer.from(svg(true))).resize(size, size).png().toFile(new URL(name, publicDir).pathname);
}
// Embed display-sized exports so a tiny header does not fetch a 1 MB SVG.
for (const [name, raster, width, height] of [['fly-logo.svg', 'fly-logo.png', 384, 280], ['favicon.svg', 'fly-icon-96.png', 96, 96]]) {
  const bytes = await readFile(new URL(raster, publicDir));
  await writeFile(new URL(name, publicDir), `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}"><image width="${width}" height="${height}" href="data:image/png;base64,${bytes.toString('base64')}" style="image-rendering:pixelated"/></svg>\n`);
}

// ICO supports embedded PNGs, preserving the same artwork at every size.
const sizes = [16, 24, 32, 48, 64, 128, 256];
const frames = await Promise.all(sizes.map(size => sharp(Buffer.from(svg(true))).resize(size, size).png().toBuffer()));
const header = Buffer.alloc(6 + sizes.length * 16);
header.writeUInt16LE(1, 2);
header.writeUInt16LE(sizes.length, 4);
let offset = header.length;
frames.forEach((frame, i) => {
  const entry = 6 + i * 16;
  header[entry] = header[entry + 1] = sizes[i] === 256 ? 0 : sizes[i];
  header.writeUInt16LE(1, entry + 4);
  header.writeUInt16LE(32, entry + 6);
  header.writeUInt32LE(frame.length, entry + 8);
  header.writeUInt32LE(offset, entry + 12);
  offset += frame.length;
});
await writeFile(new URL('favicon.ico', publicDir), Buffer.concat([header, ...frames]));
console.log('Exported the selected fly as the site logo and browser icons.');
