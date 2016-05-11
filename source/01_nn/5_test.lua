
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'

------------------------------
-- function
------------------------------

function test()

	valid_score, model_idx = vr_mean:max(1)
	local model_idx = model_idx[{1,1}]
	local valid_score = valid_score[{1,1}]

	local test_nrow = test_data:size(1)

	local vamodels = {}
	for v = 1,opt.va do
		vamodels[v] = torch.load(paths.concat(opt.save_models, 'model_va' .. v .. '_ep' .. model_idx .. '.net'))

		-- set model to evaluate mode
		vamodels[v]:cuda()
		vamodels[v]:evaluate()
	end

	path = "../../submission/01_nn/01_nn_itr".. globalItr .. "_valid" .. string.format("%.4f", valid_score) .. ".csv"
	os.execute('mkdir -p ' .. sys.dirname(path))
	local fp = io.open(path, "w")

	headers = {"ID", "TARGET"}
	headwrite = table.concat(headers, ",")
	fp:write(headwrite.."\n")

	-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')
		for t = 1,test_nrow do
		-- disp progress
		xlua.progress(t, test_nrow)

		-- get new sample
		local input = test_data[{{t},}]
		input = input:cuda()

		-- test sample
		local pred = 0
		for v = 1,opt.va do
			p = vamodels[v]:forward(input)[{1,1}]
			pred = pred + p
		end
		pred = pred / opt.va

		row = {test_id[t]}
		table.insert(row, tostring(pred))
		rowwrite = table.concat(row, ",")
		fp:write(rowwrite.."\n")
	end

	fp:close()
end
