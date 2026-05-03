function Data = my_read(files, t, n)%t为每一次扫描大小，files为单接收器接收的信号文件名构成的元组,n为探头个数
if t>40000
    Data = zeros(n, n, 40000);
else
    Data = zeros(n, n, t);
end
for i = 1:n
    for j = 1:n
        load(files{i, j}, 'C1_data');
        if t>40000
            Data(i, j, :) = C1_data(1:40000);
        else
            Data(i, j, :) = C1_data;
        end
    end
end