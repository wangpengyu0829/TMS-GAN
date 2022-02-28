clear all;
close all;
clc;
tic

% fid=fopen(['A.txt'],'w');%写入文件路径
% for i=1:150
% fprintf(fid,'Input_(%.1d)_1.png\r\n',i);   %按列输出，若要按行输出：fprintf(fid,'%.4\t',A(jj)); 
% end
% fclose(fid);

input_path1 = '.\gt\'; 
input_path2 = '.\hazy\'; 
output_path = '.\hazy2\';         
ext = '*.png';
dis = dir([input_path1 ext]);    % 获取路径下的所有图片
nms = {dis.name};                % 获取所有图片的名字
en = '.png';

% 遍历读取图片
for k = 1:length(nms)
    disp(k)
    nm1 = [input_path1 nms{k}];
    nm2 = [input_path2 nms{k}];
    I1 = imread(nm1);
    I2 = imread(nm2);
    [m1,n1,z1] = size(I1);
    I = imresize(I2, [m1 n1]);
    name = strsplit(nms{k},'.');
    sv = [output_path name{1} en];
    imwrite(I, sv);
end

toc
