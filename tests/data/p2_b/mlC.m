D='/private/tmp/claude-501/-Users-benfulcher-ClaudeCode/284548a7-cca3-4e84-b4f2-b19985958841/scratchpad/fix/p2-b';
S=load([D '/inputs.mat']); names=cellstr(S.names); R=struct();
for i=1:numel(names)
  y=double(S.(names{i})); y=y(:); r=struct();
  try, r.vgh=NW_VisibilityGraph(y,'horiz','full'); catch, r.vgh=NaN; end
  try, r.vgn=NW_VisibilityGraph(y,'norm',20000); catch, r.vgn=NaN; end
  yq=round(y*2)/2; % lattice-valued (ties, collinear points)
  try, r.vgnq=NW_VisibilityGraph(yq,'norm',20000); catch, r.vgnq=NaN; end
  try, r.vghq=NW_VisibilityGraph(yq,'horiz','full'); catch, r.vghq=NaN; end
  yr=(1:numel(y))'/numel(y)*3; % ramp: all collinear
  try, r.vgnr=NW_VisibilityGraph(yr,'norm',20000); catch, r.vgnr=NaN; end
  wk={{'biasprop',[0.1,0.5]},{'biasprop',[0.5,0.1]},{'momentum',2},{'momentum',5},{'prop',0.1},{'prop',0.5},{'prop',1.1},{'runningvar',[1.5,50]}};
  for j=1:numel(wk)
    try, r.(['walk' num2str(j)])=PH_Walker(y,wk{j}{1},wk{j}{2}); catch, r.(['walk' num2str(j)])=NaN; end
  end
  fp={{'dblwell',[1,0.2,0.1]},{'dblwell',[1,0.5,0.2]},{'dblwell',[2,0.05,0.2]},{'dblwell',[3,0.01,0.1]},{'sine',[0.5,5,0.1]},{'sine',[1,2,0.1]},{'sine',[1,5,0.1]}};
  for j=1:numel(fp)
    try, r.(['fp' num2str(j)])=PH_ForcePotential(y,fp{j}{1},fp{j}{2}); catch, r.(['fp' num2str(j)])=NaN; end
  end
  for m={'sin1','sin2','sin3'}
    try, r.(['sf_' m{1}])=SP_SinusoidFit(y,m{1}); catch, r.(['sf_' m{1}])=NaN; end
  end
  try, r.sf_sin1_short=SP_SinusoidFit(y(1:20),'sin1'); catch, r.sf_sin1_short=NaN; end
  try, r.sf_sin3_short=SP_SinusoidFit(y(1:9),'sin3'); catch, r.sf_sin3_short=NaN; end
  try, r.sf_sin3_ten=SP_SinusoidFit(y(1:10),'sin3'); catch, r.sf_sin3_ten=NaN; end
  try, r.sfD=DN_SimpleFit(y,'sin1'); catch, r.sfD=NaN; end
  % PP_Compare on raw (non-z-scored) series
  yraw=double(S.(names{i})(:))*3+2;
  for m={'sin1','sin2','medianf3','poly1','diff1'}
    try, r.(['pp_' m{1}])=PP_Compare(yraw,m{1}); catch, r.(['pp_' m{1}])=NaN; end
  end
  % SC_FluctAnal
  fa={{'nothing',[],[]},{'endptdiff',[],[]},{'range',[],[]},{'rsrange',[],[]},{'rsrangefit',1,[]},{'std',[],[]},{'iqr',[],[]},{'dfa',0,[]},{'dfa',1,[]},{'dfa',2,[]},{'dfa',3,[]}};
  for j=1:numel(fa)
    try, r.(['fa' num2str(j)])=SC_FluctAnal(y,2,fa{j}{1},50,fa{j}{2},fa{j}{3},true); catch, r.(['fa' num2str(j)])=NaN; end
  end
  try, r.fa_abs=SC_FluctAnal(zscore(abs(y)),2,'dfa',50,2,[],true); catch, r.fa_abs=NaN; end
  try, r.fa_lin=SC_FluctAnal(y,2,'dfa',50,2,[],false); catch, r.fa_lin=NaN; end
  R.(names{i})=r;
end
fid=fopen([D '/mlC.json'],'w'); fprintf(fid,'%s',jsonencode(R,'ConvertInfAndNaN',false)); fclose(fid);
disp('done')
