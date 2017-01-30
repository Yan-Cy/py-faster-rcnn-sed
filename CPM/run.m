%% Get test set

testindex = fopen('/home/chenyang/cydata/sed_subset/annodata/test.txt', 'r');
testimages = {};
testrect = {};
testcls = {};
while true
    newline = fgetl(testindex);
    if ~ischar(newline); break; end
    
    t = strsplit(newline, ' ');
    testimages = [testimages; [t{1}]];
    testcls = [testcls; [t{2}]]
    
    t1 = str2num(t{3});
    t2 = str2num(t{4});
    t3 = str2num(t{5});
    t4 = str2num(t{6});
    testrect = [testrect; [t1, t2, t3-t1, t4-t2]];
end

fclose(testindex)

%% Get train set

trainindex = fopen('/home/chenyang/cydata/sed_subset/annodata/train.txt', 'r')
trainimages = {};
trainrect = {};
while true
    newline = fgetl(trainindex);
    if ~ischar(newline); break; end
    
    t = strsplit(newline, ' ');
    trainimages = [trainimages; [t{1}]];
    traincls = [traincls; [t{2}]]
    
    t1 = str2num(t{3});
    t2 = str2num(t{4});
    t3 = str2num(t{5});
    t4 = str2num(t{6});
    trainrect = [trainrect; [t1, t2, t3-t1, t4-t2]];
end

fclose(trainindex);

%% test & train

trainresults = run_CPM(trainimages, trainrect);
testresults = run_CPM(testimages, testrect);

for  i = 1:length(trainresults)
    imgfile = fopen([trainimages{i}, '_', traincls{i}, '_', num2str(trainrect{i}(1)), '_', num2str(trainrect{i}(2)), '_', num2str(trainrect{i}(3)), '_', num2str(trainrect{i}(3))], 'w');
    for j = 1:length(trainresults{i})
        fwrite(imgfile, '%d %d %s\n', trainresults{i}(j)(1))
    end
    fclose(imgfile)
end

