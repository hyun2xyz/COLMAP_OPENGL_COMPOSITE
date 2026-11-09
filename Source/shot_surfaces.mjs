// Sample the repository SPH cache at the actual shot timeline.
import * as T from 'three';
import {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';
import {createWater} from '../Water/water.mjs';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
const [out,countText,fpsText]=process.argv.slice(2);
const count=Number(countText),fps=Number(fpsText);
if(!out||!(count>0)||!(fps>0))throw Error('output count fps required');
await mkdir(out,{recursive:true});
const binary=await readFile(new URL('../Water/water-particles.bin',import.meta.url));
globalThis.fetch=async()=>new Response(binary);
const water=await createWater(T,MarchingCubes);
for(let i=0;i<count;i++){
  water.update(i/fps);const g=water.mesh.geometry,n=g.drawRange.count;
  const xyz=g.attributes.position.array,norm=g.attributes.normal.array,b=new Float32Array(n*6);
  for(let v=0;v<n;v++)for(let c=0;c<3;c++){b[v*6+c]=xyz[v*3+c]*water.mesh.scale.x;b[v*6+3+c]=norm[v*3+c];}
  await writeFile(`${out}/${String(i).padStart(3,'0')}.bin`,Buffer.from(b.buffer));
}
console.log(JSON.stringify({frames:count,fps,solver:'repo/Water/bake-water.mjs',surface:'Three.js MarchingCubes',particles:water.particles}));
