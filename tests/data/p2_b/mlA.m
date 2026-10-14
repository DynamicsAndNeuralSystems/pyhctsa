D='/private/tmp/claude-501/-Users-benfulcher-ClaudeCode/284548a7-cca3-4e84-b4f2-b19985958841/scratchpad/fix/p2-b';
S=load([D '/inputs.mat']); names=cellstr(S.names); R=struct();
dists={'norm','ev','uni','beta','rayleigh','exp','gamma','logn','wbl'};
for i=1:numel(names)
  y=double(S.(names{i})); y=y(:); r=struct();
  try, r.cv1=DN_CV(y,1); catch, r.cv1=NaN; end
  try, r.cv2=DN_CV(y,2); catch, r.cv2=NaN; end
  try, r.cv1o=DN_CV(y+3,1); catch, r.cv1o=NaN; end
  try, r.cskew=DN_CustomSkewness(y,'pearsonMode'); catch, r.cskew=NaN; end
  for nb=[5 10 21], try, r.(['hm' num2str(nb)])=DN_HistogramMode(y,nb,true,false); catch, r.(['hm' num2str(nb)])=NaN; end, end
  try, r.hmauto=DN_HistogramMode(y,'auto'); catch, r.hmauto=NaN; end
  try, r.hmsqrt=DN_HistogramMode(y,'sqrt'); catch, r.hmsqrt=NaN; end
  try, r.hmfd=DN_HistogramMode(y,'fd'); catch, r.hmfd=NaN; end
  try, r.hmabs=DN_HistogramMode(abs(y),10,true,false); catch, r.hmabs=NaN; end
  try, r.hmhc=DN_HistogramMode(y,10,false,false); catch, r.hmhc=NaN; end
  try, r.fks=DN_FitKernelSmooth(y,'numcross',[0.05,0.1,0.2,0.3,0.4,0.5],'area',[0.05,0.1,0.2,0.3,0.4,0.5],'arclength',[0.1,0.5,1,2]); catch, r.fks=NaN; end
  for d=1:numel(dists)
    try, r.(['ks_' dists{d}])=DN_CompareKSFit(y,dists{d}); catch, r.(['ks_' dists{d}])=NaN; end
  end
  try, r.ti5=DN_TailIndex(y,0.05); catch, r.ti5=NaN; end
  try, r.ti10=DN_TailIndex(y,0.10); catch, r.ti10=NaN; end
  R.(names{i})=r;
end
fid=fopen([D '/mlA.json'],'w'); fprintf(fid,'%s',jsonencode(R,'ConvertInfAndNaN',false)); fclose(fid);
disp('done')
