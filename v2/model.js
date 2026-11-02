// Synthetic CBCT patient. One analytic model drives the 3D meshes, the voxel volume and every slice.
// Coordinates in mm: +x patient left, +y up, +z anterior. Occlusal plane at y = 0.

export const clamp=(v,a,b)=>v<a?a:v>b?b:v;
export const smin=(a,b,k)=>{const h=Math.max(k-Math.abs(a-b),0)/k;return Math.min(a,b)-h*h*k*.25};
export const smax=(a,b,k)=>-smin(-a,-b,k);
export function sdEll(x,y,z,c,r){const px=(x-c[0])/r[0],py=(y-c[1])/r[1],pz=(z-c[2])/r[2];const k0=Math.hypot(px,py,pz),k1=Math.hypot(px/r[0],py/r[1],pz/r[2]);return k0*(k0-1)/(k1||1e-6)}
export function sdRBox(x,y,z,c,h,r){const qx=Math.abs(x-c[0])-h[0]+r,qy=Math.abs(y-c[1])-h[1]+r,qz=Math.abs(z-c[2])-h[2]+r;return Math.hypot(Math.max(qx,0),Math.max(qy,0),Math.max(qz,0))+Math.min(Math.max(qx,qy,qz),0)-r}
export function sdCap(x,y,z,a,b,r){const bx=b[0]-a[0],by=b[1]-a[1],bz=b[2]-a[2],px=x-a[0],py=y-a[1],pz=z-a[2];const t=clamp((px*bx+py*by+pz*bz)/(bx*bx+by*by+bz*bz),0,1);return Math.hypot(px-bx*t,py-by*t,pz-bz*t)-r}
function sdCone(x,y,z,a,b,ra,rb){const bx=b[0]-a[0],by=b[1]-a[1],bz=b[2]-a[2],px=x-a[0],py=y-a[1],pz=z-a[2];const t=clamp((px*bx+py*by+pz*bz)/(bx*bx+by*by+bz*bz),0,1);return Math.hypot(px-bx*t,py-by*t,pz-bz*t)-(ra+(rb-ra)*t)}
export function polyDist(P,x,y,z){let m=1e9;for(let i=0;i<P.length-3;i+=3){const ax=P[i],ay=P[i+1],az=P[i+2],dx=P[i+3]-ax,dy=P[i+4]-ay,dz=P[i+5]-az;const t=clamp(((x-ax)*dx+(y-ay)*dy+(z-az)*dz)/(dx*dx+dy*dy+dz*dz),0,1);const ex=x-ax-dx*t,ey=y-ay-dy*t,ez=z-az-dz*t,d=ex*ex+ey*ey+ez*ez;if(d<m)m=d}return Math.sqrt(m)}
const v=(x,y,z)=>({x,y,z}),add=(a,b)=>v(a.x+b.x,a.y+b.y,a.z+b.z),sc=(a,s)=>v(a.x*s,a.y*s,a.z*s),madd=(a,b,s)=>v(a.x+b.x*s,a.y+b.y*s,a.z+b.z*s);
const dot=(a,b)=>a.x*b.x+a.y*b.y+a.z*b.z,cross=(a,b)=>v(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x),norm=a=>sc(a,1/Math.hypot(a.x,a.y,a.z));
const arr=p=>[p.x,p.y,p.z];

/* ── field of view ── */
export const CROP_R=42,Y0=-37,Y1=58;
export const RANGE={x:[-44,44],y:[Y0,Y1],z:[-34,42]};
export const VOL={x0:-44,y0:Y0,z0:-34,s:.5,nx:177,ny:191,nz:153};   // index = i + nx*(k + nz*j)

/* ── dental arch (2D distance lookup) ── */
export const A=25,B=30,ZC=-6,TH=1.42;
const ARCH=[];for(let i=0;i<=64;i++){const t=-1.5+3*i/64;ARCH.push(A*Math.sin(t),B*Math.cos(t)+ZC)}
function archDistRaw(x,z){let m=1e9;for(let i=0;i<128;i+=2){const ax=ARCH[i],az=ARCH[i+1],dx=ARCH[i+2]-ax,dz=ARCH[i+3]-az;const t=clamp(((x-ax)*dx+(z-az)*dz)/(dx*dx+dz*dz),0,1);const ex=x-ax-dx*t,ez=z-az-dz*t,d=ex*ex+ez*ez;if(d<m)m=d}return Math.sqrt(m)}
const LX0=-48,LZ0=-40,LS=.5,LNX=193,LNZ=181,LUT=new Float32Array(LNX*LNZ);
for(let k=0;k<LNZ;k++)for(let i=0;i<LNX;i++)LUT[i+k*LNX]=archDistRaw(LX0+i*LS,LZ0+k*LS);
function archDist(x,z){const fx=clamp((x-LX0)/LS,0,LNX-1.001),fz=clamp((z-LZ0)/LS,0,LNZ-1.001);const i=fx|0,k=fz|0,u=fx-i,w=fz-k,o=i+k*LNX;return (LUT[o]*(1-u)+LUT[o+1]*u)*(1-w)+(LUT[o+LNX]*(1-u)+LUT[o+LNX+1]*u)*w}
export const arch3=(th,y)=>[A*Math.sin(th),y,B*Math.cos(th)+ZC];
const archOut=th=>{const x=A*Math.sin(th),z=B*Math.cos(th);const n=Math.hypot(x/(A*A),z/(B*B));return[x/(A*A)/n,0,z/(B*B)/n]};

/* ── anatomy ── */
export const HEAD={c:[0,8,-2],r:[52,68,40]};
const MID={c:[0,36,-4],r:[38,30,26]},HOLLOW={c:[0,38,-30],r:[34,22,18]};
export const SINUS=[-1,1].map(s=>({c:[s*15,32,0],r:[8,10,10],c2:[s*14,25,2],r2:[5.5,4,6.5]}));
export const MUC={c:[15,23.5,2],r:[6.5,2.6,7.5]};
export const NASAL={c:[0,36,4],h:[5.5,14,13],r:3},APERT={c:[0,30,21],h:[8.5,7,6],r:5},APERT2={c:[0,40,21],h:[5,6.5,6],r:3.5};
export const ORBITS=[-1,1].map(s=>({c:[s*17.5,56,10],r:[11.5,11,16]}));
const ZYGO=[-1,1].map(s=>[[s*33,36,-2],[s*40,34,-18]]);
const TURB=[-1,1].flatMap(s=>[[[s*4.6,27,15],[s*4.6,27.5,-8],2.2],[[s*4.2,34.5,11],[s*4.2,35,-6],1.6]]);
export const AIRW=[[0,-38,-26],[0,18,-24],[0,32,-11]];
const RAMI=[-1,1].map(s=>({plate:[[s*25,-25,-7],[s*27.5,4,-10.5]],cor:[[s*26,-6,-2.5],[s*28,17,-4.5]],post:[[s*24,-29,-13],[s*29,19,-16.5]],cond:[s*30,23,-17]}));
export const CANALS=[-1,1].map(s=>[...arch3(s*.62,-20),...arch3(s*.85,-24),...arch3(s*1.15,-25),...arch3(s*1.42,-22),s*26.8,-8,-9,s*28.2,8,-12.5]);
const FORAMEN=[-1,1].map(s=>{const p=arch3(s*.62,-20),o=archOut(s*.62);return[p[0]+o[0]*3.6,p[1],p[2]+o[2]*3.6]});
export const sinusD=(s,x,y,z)=>smin(sdEll(x,y,z,s.c,s.r),sdEll(x,y,z,s.c2,s.r2),3);
export const nasalD=(x,y,z)=>Math.min(sdRBox(x,y,z,NASAL.c,NASAL.h,NASAL.r),sdRBox(x,y,z,APERT.c,APERT.h,APERT.r),sdRBox(x,y,z,APERT2.c,APERT2.h,APERT2.r));
export const airwayD=(x,y,z)=>Math.min(sdCap(x,y,z,AIRW[0],AIRW[1],6.5),sdCap(x,y,z,AIRW[1],AIRW[2],5.5));
export const canalD=(x,y,z)=>Math.min(polyDist(CANALS[0],x,y,z),polyDist(CANALS[1],x,y,z));

/* ── teeth (FDI) ── */
const W_U=[8.6,6.6,7.6,7,6.8,10.2,9.2,8.6],W_L=[5.4,5.9,7,7,7.2,11,10.5,10];
const BL_U=[.82,.9,1.05,1.3,1.3,1.1,1.12,1.12],BL_L=[1.1,1.05,1.05,1.07,1.1,.95,.95,.95];   // bucco-lingual ÷ mesio-distal
const CR_U=[10.5,9,10,8.5,8,7.5,7,6.5],RT_U=[13,13,17,14,14,13,12.5,11];
const CR_L=[9,9,10,8.5,8,7.5,7,6.5],RT_L=[12,12.5,15,14,14,13,12.5,11];
export const MISSING=new Set([18,28,48]),RESTOR=new Set([14,25,37,46]);
export const TNAME=['central incisor','lateral incisor','canine','first premolar','second premolar','first molar','second molar','third molar'];
export const QNAME={1:'Maxillary right',2:'Maxillary left',3:'Mandibular left',4:'Mandibular right'};
export const TEETH=[],TBY={};
for(const q of [1,2,3,4]){
  const up=q<3,side=(q===1||q===4)?-1:1,W=up?W_U:W_L,tot=W.reduce((a,b)=>a+b);let cum=0;
  for(let i=0;i<8;i++){
    const num=q*10+i+1,th=side*TH*(cum+W[i]/2)/tot;cum+=W[i];
    if(MISSING.has(num))continue;
    const ax=A*Math.sin(th),az=B*Math.cos(th)+ZC;
    const out=norm(v(ax/(A*A),0,(az-ZC)/(B*B))),mes=sc(norm(v(A*Math.cos(th),0,-B*Math.sin(th))),-Math.sign(th));
    const dir=up?1:-1,crown=(up?CR_U:CR_L)[i],root=(up?RT_U:RT_L)[i],lean=i<3?.2:.05;
    let tip=v(ax,dir*.6,az),axis=norm(v(-out.x*lean,dir,-out.z*lean));
    if(num===38){tip=v(ax-.5,-9,az-1.5);axis=norm(madd(v(0,-.55,0),mes,-.85))}               // mesioangular impaction
    const e2=norm(madd(out,axis,-dot(out,axis))),e1=cross(axis,e2);                          // right-handed basis (e1, axis, e2)
    const r0=W[i]*(!up&&i<2?.5:.46),multi=i>=5,len=crown+root;
    const T={num,up,i,q,th,tip,axis,e1,e2,out,mes,crown,len,r0,sm:1,so:(up?BL_U:BL_L)[i],restor:RESTOR.has(num),multi};
    T.capH=r0*(i<3?.9:.5);T.body=multi?crown+2.6:len;
    T.roots=null;
    if(multi){
      const offs=up?[[.42,.45],[-.42,.45],[0,-.55]]:[[.5,0],[-.5,0]];
      T.roots=offs.map(([om,oo])=>{
        const a=madd(madd(madd(tip,axis,crown+1.2),e1,om*r0*T.sm),e2,oo*r0*T.so);
        const b=madd(madd(madd(tip,axis,len),e1,om*r0*1.25),e2,oo*r0*1.25);
        return{a:arr(a),b:arr(b),ra:r0*(up&&oo<0?.48:.42),rb:r0*.1};
      });
    }
    T.apex=T.roots?T.roots[0].b:arr(madd(tip,axis,len));
    T.c=madd(tip,axis,len/2);T.br2=(len/2+r0*1.4+2)**2;
    TEETH.push(T);TBY[num]=T;
  }
}
// crown/trunk radius profile along the tooth axis (circular, then scaled sm/so)
export function rad(T,h){
  const r=T.r0,C=T.crown,L=T.body;
  if(h<=0||h>=L)return 0;
  if(h<T.capH){const u=1-h/T.capH;return r*Math.sqrt(Math.max(0,1-u*u))}
  if(h<C*.7)return r;
  if(h<C)return r*(1-.18*(h-C*.7)/(C*.3));
  const u=(h-C)/(L-C);
  if(T.multi)return r*.82*Math.sqrt(Math.max(0,1-u*u));
  return r*.82*(1-.62*u)*(u>.9?Math.sqrt((1-u)/.1):1);
}
export const TS={h:0};
export function toothSD(T,x,y,z){
  const dx=x-T.tip.x,dy=y-T.tip.y,dz=z-T.tip.z,a=T.axis,h=dx*a.x+dy*a.y+dz*a.z;
  const rx=dx-a.x*h,ry=dy-a.y*h,rz=dz-a.z*h;
  const rm=(rx*T.e1.x+ry*T.e1.y+rz*T.e1.z)/T.sm,ro=(rx*T.e2.x+ry*T.e2.y+rz*T.e2.z)/T.so;
  TS.h=h;
  let d=Math.max(Math.hypot(rm,ro)-rad(T,clamp(h,0,T.body)),-h,h-T.body);
  if(T.roots)for(const R of T.roots){const e=sdCone(x,y,z,R.a,R.b,R.ra,R.rb);if(e<d)d=e}
  return d;
}
// surface point of the crown at height h in a direction (b,l,m,d)
function surf(T,h,s){
  const dv=s==='b'?T.out:s==='l'?sc(T.out,-1):s==='m'?T.mes:sc(T.mes,-1);
  const k=Math.abs(dot(dv,T.e1))*T.sm+Math.abs(dot(dv,T.e2))*T.so;
  return{dv,r:rad(T,h)*k};
}

/* ── findings (colours from the Second Opinion DS) ── */
export const TYPES={
  calc:{name:'Calculus',c:'#00FFE0'},
  cinc:{name:'Caries, Incipient',c:'#FFB52D'},
  cprog:{name:'Caries, Progress',c:'#FB19DA'},
  impact:{name:'Impaction',c:'#EA6774'},
  margin:{name:'Notable Margin',c:'#7B42F6'},
  pa:{name:'Periapical Radiolucency',c:'#36CFFA'},
};
export const FIND=[
  {t:'calc',tooth:16,h:.95,s:'d'},{t:'calc',tooth:26,h:.95,s:'b'},{t:'calc',tooth:41,h:.95,s:'l'},
  {t:'cinc',tooth:24,h:.4,s:'m'},
  {t:'cprog',tooth:21,h:.3,s:'m'},{t:'cprog',tooth:36,h:0,s:'o'},{t:'cprog',tooth:45,h:.35,s:'d'},
  {t:'impact',tooth:38},
  {t:'margin',tooth:14,h:.85,s:'b'},
  {t:'pa',tooth:36},{t:'pa',tooth:12},
];
const FIND_SPEC=FIND.map(F=>({...F}));
function placeFindings(){
  FIND.length=0;
  for(const spec of FIND_SPEC){
    const T=TBY[spec.tooth];if(!T)continue;
    const F={...spec,i:FIND.length,kind:'blob'};
    if(F.t==='pa'){F.kind='lesion';const ap=T.apex;F.pos=v(ap[0]+T.axis.x*1.2,ap[1]+T.axis.y*1.2,ap[2]+T.axis.z*1.2);F.r=3.3}
    else if(F.t==='impact'){F.kind='tooth';F.pos=T.c;F.r=T.len/2}
    else if(F.s==='o'){F.pos=madd(T.tip,T.axis,1.4);F.r=2.6}
    else{const h=Math.max(T.crown*F.h,T.capH*.8),{dv,r}=surf(T,h,F.s);
      const depth=F.t==='calc'?1.05:F.t==='margin'?.95:F.t==='cinc'?.75:.5;
      F.pos=madd(madd(T.tip,T.axis,h),dv,r*depth);
      F.r=F.t==='calc'?1.5:F.t==='cprog'?2.6:F.t==='margin'?1.7:1.7}
    FIND.push(F);
  }
}
placeFindings();

/* ── bone ── */
export const P={mx:0,md:0};
export function boneParts(x,y,z){
  const crop=Math.max(Math.hypot(x,z)-CROP_R,y-Y1,Y0-y);
  const da=archDist(x,z);
  let rU=1e9,rL=1e9;                                                        // root eminences
  for(const T of TEETH){const dx=x-T.c.x,dy=y-T.c.y,dz=z-T.c.z;if(dx*dx+dy*dy+dz*dz>T.br2)continue;
    const d=smax(toothSD(T,x,y,z)-1.5,T.crown+.8-TS.h,1);if(T.up)rU=Math.min(rU,d);else rL=Math.min(rL,d)}
  let mx=1e9;
  if(y>-2){
    mx=smax(smax(da-1.8,10-y,2),y-21,5);                                     // alveolar process
    mx=smin(mx,rU,2.5);
    mx=smin(mx,smax(smax(sdEll(x,y,z,MID.c,MID.r),14-y,4),-sdEll(x,y,z,HOLLOW.c,HOLLOW.r),4),7);   // midface
    mx=smin(mx,smax(smax((Math.hypot(x/A,(z-ZC)/B)-1)*A-1,15.5-y,2),y-21,2),3);   // palate
    if(y>24){
      for(const Z of ZYGO)mx=smin(mx,sdCap(x,y,z,Z[0],Z[1],3.2),4);         // zygomatic arches
      for(const O of ORBITS)mx=Math.max(mx,-sdEll(x,y,z,O.c,O.r));
    }
    for(const s of SINUS)mx=Math.max(mx,-sinusD(s,x,y,z));
    const nas=nasalD(x,y,z);
    mx=Math.max(mx,-nas);
    mx=Math.min(mx,Math.max(Math.abs(x)-.8,nas-.4,22-y,z-17,y-50));         // septum
    if(nas<1)for(const t of TURB)mx=Math.min(mx,sdCap(x,y,z,t[0],t[1],t[2])); // turbinates
    mx=Math.max(mx,-airwayD(x,y,z));
  }
  let md=1e9;
  if(y<30){
    const bot=-31+4*Math.min(1,Math.abs(x)/24);
    md=smax(smax(da-2.2-.26*Math.max(0,-y-11),y+10,2),bot-y,4);
    md=smin(md,rL,2.5);
    md=smin(md,sdEll(x,y,z,[0,-27.5,25.5],[9,5,4]),4);                      // mental protuberance
    for(const R of RAMI){
      let ra=sdCap(x,y,z,R.plate[0],R.plate[1],5.2);
      ra=smin(ra,sdCap(x,y,z,R.cor[0],R.cor[1],3),3);                        // coronoid
      ra=smin(ra,smin(sdCap(x,y,z,R.post[0],R.post[1],3.8),sdEll(x,y,z,R.cond,[6.5,4,4.5]),3),3);   // posterior border + condyle
      md=smin(md,ra,5);
    }
    md=Math.max(md,-(canalD(x,y,z)-1.4));
    for(const f of FORAMEN)md=Math.max(md,-(Math.hypot(x-f[0],y-f[1],z-f[2])-1.7));
  }
  P.mx=Math.max(mx,crop);P.md=Math.max(md,crop);
}

/* ── sampler: grey value, structure label, finding index ── */
// labels: 0 air outside head · 1 soft tissue · 2 maxilla · 3 mandible · 4 sinus · 5 nasal · 6 airway · 7 canal · 100+i tooth · 254 outside FOV
export const SMP={g:0,lab:0,f:-1};

/* ── scan mode: a real CBCT prepared by tools/prepare_scan.py replaces the procedural model ── */
export const MODE={scan:null};
let SG=null,SL=null,SF=null;
export function useScan(meta,G,L){
  const [nx,ny,nz]=meta.dims,s=meta.spacing,[x0,y0,z0]=meta.origin;
  Object.assign(VOL,{x0,y0,z0,s,nx,ny,nz});
  RANGE.x=[x0,x0+(nx-1)*s];RANGE.y=[y0,y0+(ny-1)*s];RANGE.z=[z0,z0+(nz-1)*s];
  TEETH.length=0;for(const k in TBY)delete TBY[k];
  const mid=meta.teeth.reduce((a,t)=>[a[0]+t.centroid[0]/meta.teeth.length,a[1]+t.centroid[2]/meta.teeth.length],[0,0]);
  for(const t of meta.teeth){
    const q=Math.floor(t.fdi/10),i=t.fdi%10-1,tip=v(...t.tip),c=v(...t.centroid);
    const axis=norm(v(t.apex[0]-t.tip[0],t.apex[1]-t.tip[1],t.apex[2]-t.tip[2]));
    let out=v(c.x-mid[0],0,c.z-mid[1]);out=Math.hypot(out.x,out.z)>.1?norm(out):v(0,0,1);
    const mes=sc(norm(v(-out.z,0,out.x)),Math.sign(-c.x*(-out.z))||1);           // horizontal, towards the midline
    const e2=norm(madd(out,axis,-dot(out,axis))),e1=cross(axis,e2),len=Math.max(4,t.length),r0=Math.max(1.5,t.radius);
    const T={num:t.fdi,up:q<3,i,q,th:0,tip,axis,e1,e2,out,mes,crown:len*.42,len,body:len,r0,capH:r0*.6,sm:1,so:1,
      restor:!!t.restored,multi:false,roots:null,apex:t.apex,c,br2:(len/2+r0+2)**2,label:t.label};
    TEETH[t.label-100]=T;TBY[t.fdi]=T;
  }
  placeFindings();
  // stamp findings into a finding-index volume
  const N=nx*ny*nz,F=new Uint8Array(N).fill(255);
  for(const f of FIND){
    if(f.kind==='tooth'){const lab=TBY[f.tooth].label;for(let o=0;o<N;o++)if(L[o]===lab)F[o]=f.i;continue}
    const r=f.r,i0=Math.max(0,Math.floor((f.pos.x-r-x0)/s)),i1=Math.min(nx-1,Math.ceil((f.pos.x+r-x0)/s));
    const j0=Math.max(0,Math.floor((f.pos.y-r-y0)/s)),j1=Math.min(ny-1,Math.ceil((f.pos.y+r-y0)/s));
    const k0=Math.max(0,Math.floor((f.pos.z-r-z0)/s)),k1=Math.min(nz-1,Math.ceil((f.pos.z+r-z0)/s));
    for(let j=j0;j<=j1;j++)for(let k=k0;k<=k1;k++)for(let i=i0;i<=i1;i++){
      const dx=x0+i*s-f.pos.x,dy=y0+j*s-f.pos.y,dz=z0+k*s-f.pos.z;
      if(dx*dx+dy*dy+dz*dz<r*r)F[i+nx*(k+nz*j)]=f.i;
    }
  }
  SG=G;SL=L;SF=F;MODE.scan=meta;
  return F;
}
function sampleScan(x,y,z){
  const {x0,y0,z0,s,nx,ny,nz}=VOL,i=Math.round((x-x0)/s),j=Math.round((y-y0)/s),k=Math.round((z-z0)/s);
  if(i<0||j<0||k<0||i>=nx||j>=ny||k>=nz){SMP.g=0;SMP.lab=254;SMP.f=-1;return}
  const o=i+nx*(k+nz*j);SMP.g=SG[o];SMP.lab=SL[o];SMP.f=SF[o]===255?-1:SF[o];
}
// coarse bone field (1 mm, from the mesh pass) lets the volume skip bone maths far from bone
let CF1=null,CF2=null;
export function setCoarse(f1,f2){CF1=f1;CF2=f2}
function coarseFar(x,y,z){
  if(!CF1)return false;
  const i=Math.round((x-BONE.x0)/BONE.s),j=Math.round((y-BONE.y0)/BONE.s),k=Math.round((z-BONE.z0)/BONE.s);
  if(i<1||j<1||k<1||i>=BONE.nx-1||j>=BONE.ny-1||k>=BONE.nz-1)return false;
  const o=i+BONE.nx*(k+BONE.nz*j);return CF1[o]>2.5&&CF2[o]>2.5;
}
export function sample(x,y,z){
  if(MODE.scan)return sampleScan(x,y,z);
  SMP.f=-1;SMP.lab=0;
  if(x*x+z*z>CROP_R*CROP_R||y>Y1||y<Y0){SMP.g=0;SMP.lab=254;return}
  if(sdEll(x,y,z,HEAD.c,HEAD.r)>0){SMP.g=4;return}
  let g=78;SMP.lab=1;
  if(airwayD(x,y,z)<0){g=8;SMP.lab=6}
  for(const s of SINUS)if(sinusD(s,x,y,z)<0){g=10;SMP.lab=4;if(s.c[0]>0&&sdEll(x,y,z,MUC.c,MUC.r)<0)g=62}
  if(nasalD(x,y,z)<0){g=10;SMP.lab=5}
  for(const O of ORBITS)if(sdEll(x,y,z,O.c,O.r)<0)g=y>52?40:60;            // orbital fat
  if(coarseFar(x,y,z)){P.mx=P.md=9}else boneParts(x,y,z);
  const b=Math.min(P.mx,P.md);
  if(b<0){g=b>-1.6?222:150;SMP.lab=P.mx<P.md?2:3}
  if(SMP.lab===3&&canalD(x,y,z)<1.4){g=70;SMP.lab=7}
  for(let i=0;i<TEETH.length;i++){
    const T=TEETH[i],dx=x-T.c.x,dy=y-T.c.y,dz=z-T.c.z;
    if(dx*dx+dy*dy+dz*dz>T.br2)continue;
    const d=toothSD(T,x,y,z);
    if(d<0){
      const h=TS.h;
      g=h<T.crown?(d>-1.1?252:212):(d>-.9?205:188);
      if(d<-T.r0*.62&&h>T.crown*.55&&h<T.body-1)g=96;                       // pulp
      if(T.restor&&h<T.crown*.55)g=255;
      SMP.lab=100+i;
    }else if(d<.55&&SMP.lab<=3&&TS.h>T.crown)g=58;                          // PDL space
  }
  for(const F of FIND){
    if(F.kind==='tooth'){if(SMP.lab===100+TEETH.indexOf(TBY[F.tooth]))SMP.f=F.i;continue}
    const dx=x-F.pos.x,dy=y-F.pos.y,dz=z-F.pos.z;
    if(dx*dx+dy*dy+dz*dz<F.r*F.r){
      if(F.kind==='lesion')g=Math.min(g,64);else if(F.t==='calc')g=240;else if(F.t!=='margin')g=Math.min(g,118);
      SMP.f=F.i;
    }
  }
  SMP.g=g;
}
export const LABEL_NAME=['','Soft tissue','Maxilla','Mandible','Maxillary sinus','Nasal cavity','Airway','Mandibular canal','Restoration'];

/* ── mesh grid for bone ── */
export const BONE={x0:-45,y0:-39,z0:-36,s:1,nx:91,ny:100,nz:81};
