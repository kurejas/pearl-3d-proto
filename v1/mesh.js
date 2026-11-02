// Surface meshes from a label volume (scan mode): crop → optional downsample → blur → surface nets.
// Volume layout matches VOL: index = i + nx*(k + nz*j); field < 0 inside.

export function surfaceNetsRaw(f,nx,ny,nz,o,h){
  const id=(i,j,k)=>i+nx*(k+nz*j),cell=new Int32Array(nx*ny*nz).fill(-1),pos=[],ind=[];
  const C=[[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]];
  const E=[[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
  const v=new Float32Array(8);
  for(let j=0;j<ny-1;j++)for(let k=0;k<nz-1;k++)for(let i=0;i<nx-1;i++){
    let m=0;for(let c=0;c<8;c++){const q=C[c],val=f[id(i+q[0],j+q[1],k+q[2])];v[c]=val;if(val<0)m|=1<<c}
    if(m===0||m===255)continue;
    let sx=0,sy=0,sz=0,n=0;
    for(const [a,b] of E){const va=v[a],vb=v[b];if((va<0)!==(vb<0)){const t=va/(va-vb),pa=C[a],pb=C[b];sx+=pa[0]+t*(pb[0]-pa[0]);sy+=pa[1]+t*(pb[1]-pa[1]);sz+=pa[2]+t*(pb[2]-pa[2]);n++}}
    cell[id(i,j,k)]=pos.length/3;pos.push(o[0]+(i+sx/n)*h,o[1]+(j+sy/n)*h,o[2]+(k+sz/n)*h);
  }
  const quad=(a,b,c,d,fl)=>{if(a<0||b<0||c<0||d<0)return;if(fl)ind.push(a,c,b,a,d,c);else ind.push(a,b,c,a,c,d)};
  for(let k=1;k<nz-1;k++)for(let j=1;j<ny-1;j++)for(let i=1;i<nx-1;i++){
    const s=f[id(i,j,k)]<0;
    if(s!==(f[id(i+1,j,k)]<0))quad(cell[id(i,j-1,k-1)],cell[id(i,j,k-1)],cell[id(i,j,k)],cell[id(i,j-1,k)],!s);
    if(s!==(f[id(i,j+1,k)]<0))quad(cell[id(i-1,j,k-1)],cell[id(i-1,j,k)],cell[id(i,j,k)],cell[id(i,j,k-1)],!s);
    if(s!==(f[id(i,j,k+1)]<0))quad(cell[id(i-1,j-1,k)],cell[id(i,j-1,k)],cell[id(i,j,k)],cell[id(i-1,j,k)],!s);
  }
  const P=new Float32Array(pos),I=new Uint32Array(ind);
  // outward winding: the field must rise along the face normal
  let vote=0;const at=(x,y,z)=>{const i=Math.round((x-o[0])/h),j=Math.round((y-o[1])/h),k=Math.round((z-o[2])/h);
    return(i<0||j<0||k<0||i>=nx||j>=ny||k>=nz)?1:f[id(i,j,k)]};
  for(let t=0;t<Math.min(60,I.length/3);t++){
    const f3=Math.floor(t/60*I.length/3)*3,a=I[f3]*3,b=I[f3+1]*3,c=I[f3+2]*3;
    const ux=P[b]-P[a],uy=P[b+1]-P[a+1],uz=P[b+2]-P[a+2],wx=P[c]-P[a],wy=P[c+1]-P[a+1],wz=P[c+2]-P[a+2];
    let nx_=uy*wz-uz*wy,ny_=uz*wx-ux*wz,nz_=ux*wy-uy*wx;const l=Math.hypot(nx_,ny_,nz_)||1;nx_*=h/l;ny_*=h/l;nz_*=h/l;
    const cx=(P[a]+P[b]+P[c])/3,cy=(P[a+1]+P[b+1]+P[c+1])/3,cz=(P[a+2]+P[b+2]+P[c+2])/3;
    vote+=Math.sign(at(cx+nx_,cy+ny_,cz+nz_)-at(cx-nx_,cy-ny_,cz-nz_));
  }
  if(vote<0)for(let t=0;t<I.length;t+=3){const x=I[t+1];I[t+1]=I[t+2];I[t+2]=x}
  return{pos:P,ind:I};
}

// targets: [{key, labels:[codes], ds}] → [{key, pos, ind}]
export function labelMeshes(L,dims,origin,s,targets){
  const [nx,ny,nz]=dims,bb=new Int32Array(256*6);
  for(let l=0;l<256;l++){bb[l*6]=bb[l*6+2]=bb[l*6+4]=1e9;bb[l*6+1]=bb[l*6+3]=bb[l*6+5]=-1}
  for(let j=0,o=0;j<ny;j++)for(let k=0;k<nz;k++)for(let i=0;i<nx;i++,o++){
    const l=L[o]*6;if(i<bb[l])bb[l]=i;if(i>bb[l+1])bb[l+1]=i;if(j<bb[l+2])bb[l+2]=j;if(j>bb[l+3])bb[l+3]=j;if(k<bb[l+4])bb[l+4]=k;if(k>bb[l+5])bb[l+5]=k;
  }
  const out=[];
  for(const t of targets){
    const want=new Uint8Array(256);let i0=1e9,i1=-1,j0=1e9,j1=-1,k0=1e9,k1=-1;
    for(const l of t.labels){want[l]=1;if(bb[l*6+1]<0)continue;
      i0=Math.min(i0,bb[l*6]);i1=Math.max(i1,bb[l*6+1]);j0=Math.min(j0,bb[l*6+2]);j1=Math.max(j1,bb[l*6+3]);k0=Math.min(k0,bb[l*6+4]);k1=Math.max(k1,bb[l*6+5])}
    if(i1<0)continue;
    const ds=t.ds||1,pad=2;
    // cropped, downsampled occupancy with a 2-cell empty border
    const cx=Math.ceil((i1-i0+1)/ds)+pad*2,cy=Math.ceil((j1-j0+1)/ds)+pad*2,cz=Math.ceil((k1-k0+1)/ds)+pad*2;
    let g=new Float32Array(cx*cy*cz);const inv=1/(ds*ds*ds);
    for(let j=j0;j<=j1;j++)for(let k=k0;k<=k1;k++){const row=nx*(k+nz*j);for(let i=i0;i<=i1;i++)if(want[L[row+i]]){
      g[(((i-i0)/ds|0)+pad)+cx*((((k-k0)/ds|0)+pad)+cz*(((j-j0)/ds|0)+pad))]+=inv}}
    // [1 2 1]/4 blur along each axis, twice, for a smooth iso-surface
    const tmp=new Float32Array(g.length);
    for(let pass=0;pass<2;pass++)for(const [st,n] of [[1,cx],[cx,cz],[cx*cz,cy]]){
      for(let o=0;o<g.length;o++){const c=(Math.floor(o/st)%n);tmp[o]=(g[o]*2+(c>0?g[o-st]:0)+(c<n-1?g[o+st]:0))/4}
      g.set(tmp);
    }
    for(let o=0;o<g.length;o++)g[o]=.5-g[o];
    const h=s*ds,o3=[origin[0]+(i0-pad*ds+(ds-1)/2)*s,origin[1]+(j0-pad*ds+(ds-1)/2)*s,origin[2]+(k0-pad*ds+(ds-1)/2)*s];
    const m=surfaceNetsRaw(g,cx,cy,cz,o3,h);
    out.push({key:t.key,pos:m.pos,ind:m.ind});
  }
  return out;
}
