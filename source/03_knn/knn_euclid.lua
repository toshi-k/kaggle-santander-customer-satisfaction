
------------------------------
-- library
------------------------------

require 'xlua'
require 'csvigo'
require 'optim'

require 'cutorch'

require 'read_data'
require 'save_feature'

require 'lib/orderedPairs'
require 'lib/fill_na'
require 'lib/log_transform'
require 'lib/normalize_col'
require 'lib/normalize_row'

torch.setdefaulttensortype('torch.FloatTensor')

------------------------------
-- function
------------------------------

function sim_euclid(cmat, vec, cmat2)
	local cvec = vec:cuda()
	local ones = torch.ones(cmat:size(1)):cuda()
	-- local cmat2 = cmat:clone()
	cmat2:addr(1, cmat, -1, ones, cvec)
	cmat2:pow(2)
	local sim = cmat2:sum(2)
	sim:pow(0.5)
	return sim[{{},1}]:float()
end

function knn(data, label, pred_data, istrain)

	local pred = torch.Tensor(pred_data:size(1), 4):fill(0)

	local shift
	if istrain then
		shift = 1
	else
		shift = 0
	end

	ctrain_data = data:cuda()
	ctrain_data2 = ctrain_data:clone()

	for i = 1,pred_data:size(1) do

		local _, nn_index = torch.sort(sim_euclid(ctrain_data, pred_data[i], ctrain_data2), false)

		pred[{{i},{1}}] = label:index(1,nn_index[{{1+shift}}])
		pred[{{i},{2}}] = label:index(1,nn_index[{{1+shift,1+shift+4}}]):float():mean()
		pred[{{i},{3}}] = label:index(1,nn_index[{{1+shift,1+shift+8}}]):float():mean()
		pred[{{i},{4}}] = label:index(1,nn_index[{{1+shift,1+shift+26}}]):float():mean()

		if i % 100 == 0 then
			xlua.progress(i, pred_data:size(1))
			collectgarbage();
		end
	end

	return pred
end

------------------------------
-- main
------------------------------

-- load data: ----------
print('==> Load data')

train_unique_data, train_unique_label, train_unique_id = read_data("../../input/train_unique.csv", true)
collectgarbage();

train_data, train_label, train_id = read_data("../../input/train.csv", true)
collectgarbage();

test_data, _, test_id = read_data("../../input/test.csv", false)
collectgarbage();

-- preprocess (fill na): ----------
print('==> Fill NA')

mean_ = get_mean_(train_data)
train_unique_data	= fill_na(train_unique_data, mean_)
train_data 			= fill_na(train_data, mean_)
test_data			= fill_na(test_data, mean_)
collectgarbage();

-- preprocess (log transform): ----------
print('==> Log transform')

mins = get_mins(train_data)
train_unique_data 	= log_transform(train_unique_data, mins)
train_data 			= log_transform(train_data, mins)
test_data 			= log_transform(test_data, mins)
collectgarbage();

-- preprocess (normalze columns): ----------
print('==> Normalize columns')

mean_, std_ = get_mean_std(train_data)
train_unique_data 	= normalize_col(train_unique_data, mean_, std_)
train_data 			= normalize_col(train_data, mean_, std_)
test_data 			= normalize_col(test_data, mean_, std_)
collectgarbage();

-- KNN: ----------
print('==> KNN')

train_pred = knn(train_unique_data, train_unique_label, train_data, true)
test_pred = knn(train_unique_data, train_unique_label, test_data, false)

-- Save result: ----------
print('==> Save Result')

feature_names = {"ID", "KNN_euclid1", "KNN_euclid5", "KNN_euclid9", "KNN_euclid27"}
save_feature(train_id, train_pred, "../../add_features/KNN_euclid_train.csv", feature_names)
save_feature(test_id, test_pred, "../../add_features/KNN_euclid_test.csv", feature_names)
