// Computes slabs of the bone fields (for meshing) and of the voxel volume (for slicing), in parallel.
import {boneParts,P,sample,SMP,VOL,BONE,setCoarse} from './model.js';
import {labelMeshes} from './mesh.js';

onmessage=({data:job})=>{
  const {j0,j1}=job;
  if(job.kind==='field'){setCoarse(job.f1,job.f2);return}
  if(job.kind==='labelMeshes'){
    const res=labelMeshes(job.L,job.dims,job.origin,job.s,job.targets);
    postMessage({kind:'labelMeshes',meshes:res},res.flatMap(m=>[m.pos.buffer,m.ind.buffer]));
    return;
  }
  if(job.kind==='bone'){
    const {x0,y0,z0,s,nx,nz}=BONE,n=nx*nz*(j1-j0),f1=new Float32Array(n),f2=new Float32Array(n);
    let o=0;
    for(let j=j0;j<j1;j++)for(let k=0;k<nz;k++)for(let i=0;i<nx;i++,o++){
      boneParts(x0+i*s,y0+j*s,z0+k*s);f1[o]=P.mx;f2[o]=P.md;
    }
    postMessage({...job,f1,f2},[f1.buffer,f2.buffer]);
  }else{
    const {x0,y0,z0,s,nx,nz}=VOL,n=nx*nz*(j1-j0),g=new Uint8Array(n),l=new Uint8Array(n),f=new Uint8Array(n);
    let o=0;
    for(let j=j0;j<j1;j++)for(let k=0;k<nz;k++)for(let i=0;i<nx;i++,o++){
      sample(x0+i*s,y0+j*s,z0+k*s);g[o]=SMP.g;l[o]=SMP.lab;f[o]=SMP.f<0?255:SMP.f;
    }
    postMessage({...job,g,l,f},[g.buffer,l.buffer,f.buffer]);
  }
};
