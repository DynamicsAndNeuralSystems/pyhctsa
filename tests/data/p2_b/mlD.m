D='/private/tmp/claude-501/-Users-benfulcher-ClaudeCode/284548a7-cca3-4e84-b4f2-b19985958841/scratchpad/fix/p2-b';
S=load([D '/inputs.mat']); names=cellstr(S.names); R=struct();
cfg={{'AC1','ent',10,1},{'AC1','std',2,1},{'AC1','std',10,1},{'AC1','permen',10,2},{'asymAC1','ent',10,1},{'asymAC1','std',10,1},...
 {'ent','std',2,1},{'ent','std',10,1},{'lillie','std',2,1},{'mean','ent',10,1},{'mean','std',2,1},{'mean','std',10,1},{'mom3','ent',10,1},{'mom3','permen',10,2},...
 {'std','ent',10,1},{'std','std',10,1},{'permen','std',2,1},{'permen','std',5,10},{'specen','ent',5,1},{'specen','std',2,1},{'specen','std',10,1},{'specen','permen',10,2},{'specen','ent',2,1},{'mean','ent',2,1},{'specen','ent',10,1}};
for i=1:numel(names)
  y=double(S.(names{i})); y=y(:); r=struct();
  for j=1:numel(cfg)
    c=cfg{j};
    try, r.(['sw' num2str(j)])=SY_SlidingWindow(y,c{1},c{2},c{3},c{4}); catch, r.(['sw' num2str(j)])=NaN; end
  end
  try, r.sw_s_short=SY_SlidingWindow(y(1:30),'specen','std',10,1); catch, r.sw_s_short=NaN; end
  R.(names{i})=r;
end
fid=fopen([D '/mlD.json'],'w'); fprintf(fid,'%s',jsonencode(R,'ConvertInfAndNaN',false)); fclose(fid);
disp('done')
