D='/private/tmp/claude-501/-Users-benfulcher-ClaudeCode/284548a7-cca3-4e84-b4f2-b19985958841/scratchpad/fix/p2-b';
S=load([D '/inputs.mat']); names=cellstr(S.names); R=struct();
for i=1:numel(names)
  y=double(S.(names{i})); y=y(:); r=struct();
  for t={'abs','pos','neg'}
    try, r.(['oi_' t{1}])=DN_OutlierInclude(y,t{1},0.01); catch, r.(['oi_' t{1}])=NaN; end
  end
  try, r.oi_abs_thr2=DN_OutlierInclude(y,'abs',0.01,2); catch, r.oi_abs_thr2=NaN; end
  try, r.oi_neg_thr1=DN_OutlierInclude(y,'neg',0.01,1); catch, r.oi_neg_thr1=NaN; end
  for m={'gauss1','gauss2','exp1','power1'}
    for nb={'sqrt',0,15}
      key=['sf_' m{1} '_' num2str(nb{1})];
      try, r.(key)=DN_SimpleFit(y,m{1},nb{1}); catch, r.(key)=NaN; end
    end
  end
  try, r.sf_pow=DN_SimpleFit(y-min(y)+1,'power1','sqrt'); catch, r.sf_pow=NaN; end
  for tst={'runsz','runstest','lbq'}
    try, r.(['it_' tst{1}])=HT_IndependenceTests(y,tst{1}); catch, r.(['it_' tst{1}])=NaN; end
  end
  try, r.ht_runsz=HT_HypothesisTest(y,'runsz'); catch, r.ht_runsz=NaN; end
  try, r.vr1=SY_VarRatioTest(y,2,0); catch, r.vr1=NaN; end
  try, r.vr2=SY_VarRatioTest(y,[2,4,6,8,2,4,6,8],[0,0,0,0,1,1,1,1]); catch, r.vr2=NaN; end
  for nn=[50 100]
    try, r.(['dm' num2str(nn)])=SY_DriftingMean(y,'fix',nn); catch, r.(['dm' num2str(nn)])=NaN; end
  end
  for ns=[3 5]
    for eo={'each','par'}
      try, r.(['ld' num2str(ns) eo{1}])=SY_LocalDistributions(y,ns,eo{1}); catch, r.(['ld' num2str(ns) eo{1}])=NaN; end
    end
  end
  try, r.ld2=SY_LocalDistributions(y,2,'each'); catch, r.ld2=NaN; end
  try, r.ld3b=SY_LocalDistributions(y,3,'par',20); catch, r.ld3b=NaN; end
  try, r.sp_fft=SP_Summaries(y,'fft',[],[],false); catch, r.sp_fft=NaN; end
  try, r.sp_pg=SP_Summaries(y,'periodogram','hamming',[],false); catch, r.sp_pg=NaN; end
  try, r.sp_we=SP_Summaries(y,'welch','rect',[],false); catch, r.sp_we=NaN; end
  try, r.cwt_db3=WL_cwt(y,'db3',32); catch, r.cwt_db3=NaN; end
  try, r.cwt_morl=WL_cwt(y,'morl',32); catch, r.cwt_morl=NaN; end
  R.(names{i})=r;
end
fid=fopen([D '/mlB.json'],'w'); fprintf(fid,'%s',jsonencode(R,'ConvertInfAndNaN',false)); fclose(fid);
disp('done')
