
------------------------------
-- library
------------------------------

require 'torch'
require 'csvigo'

------------------------------
-- function
------------------------------

function table_count(tb)
	local count = 0
	for _ in pairs(tb) do count = count + 1 end
	return count
end

function read_data(path)
	local train = csvigo.load{path = path}

	local train_nrow = #train["ID"]
	print("data_nrow: " .. train_nrow)

	local train_ncol = table_count(train)
	print("data_ncol: " .. train_ncol)

	local train_data = torch.Tensor(train_nrow, train_ncol - 1)
	local count = 1
	for feat, ver in pairs(train) do
		if feat ~= "ID" then
			train_data[{ {},count }] = torch.Tensor(ver)
			count = count + 1
		end
	end

	train_id = torch.Tensor(train["ID"])

	train_data = train_data:float()
	train_id = train_id:int()

	return train_data, train_id
end

function read_label(path)
	local train = csvigo.load{path = path}

	local train_nrow = #train["x"]
	print("data_nrow: " .. train_nrow)

	train_label = torch.Tensor(train["x"])
	train_label = train_label:int()
	return train_label
end

------------------------------
-- main
------------------------------

-- load data: ----------

print("read train data")
train_path = "../../input/nn_train_x.csv"
train_data, train_id = read_data(train_path)

print("read train label")
label_path = "../../input/nn_train_y.csv"
train_label = read_label(label_path)

print("read test data")
test_path = "../../input/nn_test_x.csv"
test_data, test_id = read_data(test_path)

collectgarbage();

-- normalize: ----------

mins = train_data:min(1)
for i = 1,train_data:size(2) do
	train_data[{{},i}]:add(-mins[{1,i}])
	test_data[{{},i}]:add(-mins[{1,i}])

	train_data[{{},i}]:log1p()
	t = test_data[{{},i}]
	t[torch.lt(t,0)] = 0
	t:log1p()
	test_data[{{},i}] = t
end

mean_ = train_data:mean(1)
std_ = train_data:std(1)

for i = 1,train_data:size(2) do
	train_data[{{},i}]:add(-mean_[{1,i}])
	train_data[{{},i}]:div(std_[{1,i}])

	test_data[{{},i}]:add(-mean_[{1,i}])
	test_data[{{},i}]:div(std_[{1,i}])
end
