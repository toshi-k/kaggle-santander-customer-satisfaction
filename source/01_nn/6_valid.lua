
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'optim'

require 'lib/auc'

------------------------------
-- function
------------------------------

function valid()
	collectgarbage()
	print("")

	-- local vars
	local vascores = torch.Tensor(opt.va)

	-- set model to evaluate mode
	for i = 1,opt.va do
		vamodels[i]:evaluate()
	end

	-- test over test data
	print(sys.COLORS.green .. '==> validating on valid set:')
	for v = 1,opt.va do

		local loss = 0
		local va_test_nrow = vatest_data[v]:size(1)
		local va_test_pred = torch.Tensor(va_test_nrow):zero()

		for t = 1,va_test_nrow,opt.batchSize do
			-- disp progress
			xlua.progress(t, va_test_nrow)

			local local_batchSize = math.min(t+opt.batchSize-1,va_test_nrow) - t + 1
			local inputs = torch.Tensor(local_batchSize, vatest_data[v]:size(2))

			local count = 1
			for i = t,math.min(t+opt.batchSize-1,va_test_nrow) do
				-- load new sample
				inputs[{{count}}] = vatest_data[v][{{i},}]
				count = count + 1
			end

			inputs = inputs:cuda()
			-- test sample
			local pred = vamodels[v]:forward(inputs)
			pred = pred:view(-1):float()
			va_test_pred[{{t,math.min(t+opt.batchSize-1,va_test_nrow)}}] = pred
		end
		xlua.progress(va_test_nrow, va_test_nrow)

		vascores[v] = auc(vatest_label[v], va_test_pred)
		print("va_score[" .. v .. "]: " .. string.format("%.4f", vascores[v]))

		valid_result[{vaepoch-1, v}] = vascores[v]
	end

	va_score = vascores:mean()
	print("va_score:" .. string.format("%.4f", va_score))
end
