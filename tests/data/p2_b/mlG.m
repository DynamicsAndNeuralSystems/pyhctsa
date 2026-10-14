D='/private/tmp/claude-501/-Users-benfulcher-ClaudeCode/284548a7-cca3-4e84-b4f2-b19985958841/scratchpad/fix/p2-b';
S=load([D '/inputs.mat']); names=cellstr(S.names); R=struct();
for i=1:numel(names)
  y=double(S.(names{i})); y=y(:); r=struct();
  yr=y*30+800; % RR-interval-like (raw) scale
  try, r.raw=MD_RawHRVMeas(yr); catch, r.raw=NaN; end
  try, r.hrv=MD_hrv_classic(yr); catch, r.hrv=NaN; end
  try, r.porta6=MD_Porta(y,6); catch, r.porta6=NaN; end
  try, r.porta4=MD_Porta(y,4); catch, r.porta4=NaN; end
  try, r.ha10=DN_HistogramAsymmetry(y,10,false); catch, r.ha10=NaN; end
  try, r.ha11=DN_HistogramAsymmetry(y,11,false); catch, r.ha11=NaN; end
  try, r.ha11s=DN_HistogramAsymmetry(y,11,true); catch, r.ha11s=NaN; end
  try, r.lg_rand=SY_LocalGlobal(y,'randcg',100); catch, r.lg_rand=NaN; end
  try, r.lg_rand7=SY_LocalGlobal(y,'randcg',50,7); catch, r.lg_rand7=NaN; end
  try, r.srl=SY_SpreadRandomLocal(y,100,100); catch, r.srl=NaN; end
  try, r.srl_ac2=SY_SpreadRandomLocal(y,'ac2',100); catch, r.srl_ac2=NaN; end
  try, r.srl_s5=SY_SpreadRandomLocal(y,50,30,5); catch, r.srl_s5=NaN; end
  try, r.rp_rand=CO_RemovePoints(y,'random',0.1,'remove'); catch, r.rp_rand=NaN; end
  try, r.rp_rand3=CO_RemovePoints(y,'random',0.3,'remove',3); catch, r.rp_rand3=NaN; end
  try, yp=PP_PreProcess(y,'',[],[],false); r.rmgd=yp.rmgd(:)'; catch, r.rmgd=NaN; end
  try, yp=PP_PreProcess(y,'',[],[],false,5); r.rmgd5=yp.rmgd(:)'; catch, r.rmgd5=NaN; end
  try, r.pmf=PP_ModelFit(y,'ar',2); catch, r.pmf=NaN; end
  R.(names{i})=r;
end
fid=fopen([D '/mlG.json'],'w'); fprintf(fid,'%s',jsonencode(R,'ConvertInfAndNaN',false)); fclose(fid);
disp('done')
