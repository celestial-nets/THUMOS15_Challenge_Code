clear all;close all;clc

addpath('..');
load TEST_ANO

% ffprobe -select_streams v -show_streams input.avi 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//'

vidList_path = '/home/share/Projects/data/2015/TestData/thumos15_testing_resized/';

FrameRate = cell(0);
for i = 1:size(uniqVid, 1)
    
    vid_str = [vidList_path, uniqVid{i, 1}(1:end-3), 'avi'];
    
    filename = '/home/yuanjun/Desktop/a1s2d3f4.txt';

    cmd_str = ['ffprobe -select_streams v -show_streams ', vid_str, ...
               ' 2>/dev/null | grep avg_frame_rate | sed -e ', '''', 's/avg_frame_rate=//', ...
               '''', ' > ', filename];
           
    unix(cmd_str);
    

    formatSpec = '%s%[^\n\r]';
    delimiter = '';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
    FrameRate{i} = str2num(dataArray{1}{1});
    fclose(fileID);

    unix(['rm ', filename]);
    disp(i);
    
end