
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'cunn'

------------------------------
-- main
------------------------------

for i = 1,opt.va do
	vamodels[i]:cuda()
end
criterion:cuda()

-- Retrieve parameters and gradients:
vaparameters = {}
vagradParameters = {}

for i = 1,opt.va do
	if vamodels[i] then
		vaparameters[i], vagradParameters[i] = vamodels[i]:getParameters()
	end
end

print '==> configuring optimizer'
-- SGD
optimStates = {}
optimMethods = {}
for i = 1,opt.va do
	optimState = {
		learningRate = opt.learningRate,
		weightDecay = opt.weightDecay,
		momentum = opt.momentum,
		learningRateDecay = 1e-7
	}
	table.insert(optimStates, optimState)
	table.insert(optimMethods, optim.adam)
end

print '==> defining training procedure'

vatrain_scores = torch.Tensor(opt.va)
function train()
	collectgarbage()
	print("")

	-- epoch tracker
	vaepoch = vaepoch or 1

	-- set model to training mode
	-- model:training()
	for i = 1,opt.va do
		vamodels[i]:training()
	end

	-- do one epoch
	print(sys.COLORS.cyan .. '==> training on train set: # ' .. vaepoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for v = 1,opt.va do

		local va_train_nrow = vatrain_data[v]:size(1)

		-- shuffle at each epoch
		shuffle = torch.randperm(va_train_nrow)

		local nlloss = 0

		-- print("==> validation set: " .. v)

		for t = 1,va_train_nrow,opt.batchSize do
			-- disp progress
			xlua.progress(t, va_train_nrow)

			-- create mini batch

			local local_batchSize = math.min(t+opt.batchSize-1,va_train_nrow) - t + 1
			local inputs = torch.Tensor(local_batchSize, vatrain_data[v]:size(2))
			local targets = torch.Tensor(local_batchSize)

			local count = 1
			for i = t,math.min(t+opt.batchSize-1,va_train_nrow) do
				-- load new sample
				inputs[{{count}}] = vatrain_data[v][{{shuffle[i]},}]
				targets[{count}] = vatrain_label[v][shuffle[i]]
				count = count + 1
			end

			inputs = inputs:cuda()
			targets = targets:cuda()

			-- create closure to evaluate f(X) and df/dX
			local feval = function(x)
				-- get new parameters
				--if x ~= parameters then
				if x ~= vaparameters[v] then
					-- parameters:copy(x)
					vaparameters[v]:copy(x)
				end

				-- reset gradients
				-- gradParameters:zero()
				vagradParameters[v]:zero()

				-- evaluate function for complete mini batch
				local output = vamodels[v]:forward(inputs)
				local f = criterion:forward(output, targets)

				-- estimate df/dW
				local df_do = criterion:backward(output, targets)
				vamodels[v]:backward(inputs, df_do)

				-- update confusion
				nlloss = nlloss + f * local_batchSize

				vagradParameters[v]:div(local_batchSize)
				f = f/local_batchSize

				-- return f and df/dX
				return f,vagradParameters[v]
			end

			-- optimize on current mini-batch
			optimMethod = optimMethods[v]
			optimMethod(feval, vaparameters[v], optimStates[v])
		end
		xlua.progress(va_train_nrow, va_train_nrow)

		nlloss = nlloss / va_train_nrow
		vatrain_scores[v] = nlloss
		print("vatrain_nll[" .. v .. "]: " .. string.format("%.4f", nlloss))

		-- save/log current net
		local filename = paths.concat(opt.save_models, 'model_va' .. v .. '_ep' .. vaepoch .. '.net')
		os.execute('mkdir -p ' .. sys.dirname(filename))
		-- print('==> saving model to '.. filename)
		torch.save(filename, vamodels[v]:clearState())
	end

	-- next epoch
	vatrain_score = vatrain_scores:mean()
	print("vatrain_nll:" .. string.format("%.4f", vatrain_score))
	vaepoch = vaepoch + 1
end
