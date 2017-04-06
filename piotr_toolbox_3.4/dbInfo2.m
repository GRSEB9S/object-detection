function [pth,setIds,vidIds,dbName,skip,ext] = dbInfo2( name1, pth )
% Specifies data amount and location.
%
% 'name' specifies the name of the dataset. Valid options include: 'Usa',
% 'UsaTest', 'UsaTrain', 'InriaTrain', 'InriaTest', 'Japan', 'TudBrussels',
% 'ETH', and 'Daimler'. If dbInfo() is called without specifying the
% dataset name, defaults to the last used name (or 'UsaTest' on first call
% to dbInfo()). Finally, one can specify a subset of a dataset by appending
% digits to the end of the name (eg. 'UsaTest01' indicates first set of
% 'UsaTest' and 'UsaTest01005' indicate first set, fifth video).
%
% USAGE
%  [pth,setIds,vidIds,skip,ext] = dbInfo( [name] )
%
% INPUTS
%  name     - ['UsaTest'] specify dataset, caches last passed in name
%
% OUTPUTS
%  pth      - directory containing database
%  setIds   - integer ids of each set
%  vidIds   - [1xnSets] cell of vectors of integer ids of each video
%  skip     - specify subset of frames to use for evaluation
%  ext      - file extension determining image format ('jpg' or 'png')
%
% EXAMPLE
%  [pth,setIds,vidIds,skip,ext] = dbInfo
%
% See also
%
% Caltech Pedestrian Dataset     Version 3.2.1
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

persistent name; % cache last used name
if(nargin && ~isempty(name1)), name=lower(name1); else
if(isempty(name)), name='kaist-train-all'; end; end; name1=name;


vidId=str2double(name1(end-2:end)); % check if name ends in 3 ints

if(isnan(vidId)), vidId=[]; else name1=name1(1:end-3); end
setId=str2double(name1(end-1:end)); % check if name ends in 2 ints
if(isnan(setId)), setId=[]; else name1=name1(1:end-2); end

switch name1
  % KAIST Multispectral Pedestrian Dataset (CVPR15)
  case 'kaist-train-all'
    setIds=0:5;     subdir='kaist'; skip=2; ext='png';     % Captured at 20fps
    vidIds={0:8 0:5 0:4 0:1 0:1 0}; dbName = 'kaist';
  
  case 'kaist-test-all'
    setIds=6:11;    subdir='kaist'; skip=30; ext='png'; 
    vidIds={0:4 0:2 0:2 0 0:1 0:1}; dbName = 'kaist';
  
  case 'kaist-test-day'
    setIds=6:8;    subdir='kaist-day'; skip=30; ext='jpg';
    vidIds={0:4 0:2 0:2}; dbName = 'kaist';
  
  case 'kaist-test-night'
    setIds=9:11;    subdir='kaist-night'; skip=30; ext='jpg';
    vidIds={0 0:1 0:1}; dbName = 'kaist';
    
  % KAIST All-Day Visual Place Recognition Dataset (CVPRW15 - VPRICE))
  case 'kaist-place'
    setIds=0:5;     subdir='place'; skip=20; ext='png';     % Captured at 20fps
    vidIds={0:3 0:3 0:3 0:3 0:3 0:3 }; dbName = 'kaist-place';  
  otherwise, error('unknown data type: %s',name);
end

% optionally select only specific set/vid if name ended in ints
if(~isempty(setId)), setIds=setIds(setId); vidIds=vidIds(setId); end
if(~isempty(vidId)), vidIds={vidIds{1}(vidId)}; end

% actual directory where data is contained
if ~exist( 'pth', 'var' )
    pth=fileparts(mfilename('fullpath'));
    pth=[pth filesep 'data-' subdir];
end
end
