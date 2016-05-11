
------------------------------
-- library
------------------------------

require 'nn'
require 'cunn'

------------------------------
-- option
------------------------------

cmd = torch.CmdLine()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 8, 'number of threads')
-- paths:
cmd:option('-save_models', 'models', 'subdirectory to save/log experiments in')
-- valid:
cmd:option('-va', 5, 'num of validation-sets')
-- training:
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 32, 'mini-batch size)')
cmd:option('-weightDecay', 0, 'weight decay')
cmd:option('-momentum', 0.5, 'momentum')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(opt.seed)
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

------------------------------
-- main
------------------------------

-----

dofile "1_data.lua"

for mi=1,200 do

	vaepoch = 1

	globalItr = mi
	print("\n==========" .. globalItr .. "/" .. 200 .. "==========\n")

	-- tic:
	start_time = os.date()

	dofile "2_split.lua"
	dofile "3_model.lua"
	dofile "4_train.lua"
	dofile "5_test.lua"
	dofile "6_valid.lua"

	-----

	Itr = 30

	valid_result = torch.Tensor(Itr,opt.va):zero()

	-- print '==> training!'
	for i = 1,Itr do
		train()
		valid()
	end

	print("")
	print(valid_result)

	vr_mean = valid_result:mean(2)[{{},{1}}]
	print("max of mean: " .. string.format("%.4f", vr_mean:max()))

	vr_max = valid_result:max(1)[{{1},{}}]
	print("mean of max: " .. string.format("%.4f", vr_max:mean()))

	test()

	-----

	-- tac:
	print '===> computation time'
	end_time = os.date()
	print("start_time:\t" .. start_time)
	print("end_time:\t" .. end_time)

end
