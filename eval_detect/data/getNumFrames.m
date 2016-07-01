clear all;close all;clc

D = dir('./Thumos15_test/');
tmpInd = false(length(D), 1);
for i = 1:length(D)
    if strcmp(D(i).name, '.') || strcmp(D(i).name, '..')
        tmpInd(i) = true;
        continue;
    end
end
D(tmpInd) = [];


% frmCnt = cell(length(D), 2);
for i = 5456:length(D)
    
    str = ['ffprobe -select_streams v -show_streams ./Thumos15_test/', D(i).name, ...
           ' 2>/dev/null | grep nb_frames | sed -e ', '''', 's/nb_frames=//', '''', ...
           ' > tmp.txt'];
    unix(str);
    frmCnt{i, 1} = D(i).name;
    frmCnt{i, 2} = load('tmp.txt');
    
    unix('rm tmp.txt');
    disp(i);
    
end
% save frmCnt frmCnt
    



% ffprobe -select_streams v -show_streams input.avi 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//'
