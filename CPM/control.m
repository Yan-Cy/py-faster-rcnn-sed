function control(indexfile)

index = fopen(indexfile, 'r')
images = {};
rect = {};
while true
    newline = fgetl(index);
    if ~ischar(newline); break; end
    
    t = strsplit(newline, ' ');
    images = [images; [t{1}]];
    cls = [cls; [t{2}]]
    
    t1 = str2num(t{3});
    t2 = str2num(t{4});
    t3 = str2num(t{5});
    t4 = str2num(t{6});
    rect = [rect; [t1, t2, t3-t1, t4-t2]];
end

fclose(index);

%% test & train

results = run_CPM(images, rect);

for  i = 1:length(results)
    imgfile = fopen(['results/', images{i}, '_', cls{i}, '_', num2str(rect{i}(1)), '_', num2str(rect{i}(2)), '_', num2str(rect{i}(3)), '_', num2str(rect{i}(4))], 'w');
    for j = 1:length(results{i})
        fwrite(imgfile, '%d %d %s\n', results{i}(j)(1))
    end
    fclose(imgfile)
end

