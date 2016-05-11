
------------------------------
-- library
------------------------------

require 'torch'

------------------------------
-- option
------------------------------

if not opt then
	print '==> processing options'
	cmd = torch.CmdLine()
	cmd:text('Options:')
	-- valid:
	cmd:option('-va', 5, 'num of validation-sets')
	cmd:text()
	opt = cmd:parse(arg or {})
end

------------------------------
-- main
------------------------------

-- split valid: ----------

if opt.va > 0 then

	math.randomseed(globalItr)
	torch.manualSeed(globalItr)
	cutorch.manualSeed(globalItr)

	rate_valid = 0.1

	vatrain_data = {}
	vatrain_label = {}
	vatest_data = {}
	vatest_label = {}

	train_nrow = train_data:size(1)

	for i = 1,opt.va do
		ids = torch.randperm(train_nrow)
		vatest_data[i] = train_data:index(1,ids[{{1,math.floor(train_nrow*rate_valid)}}]:type("torch.LongTensor"))
		vatest_label[i] = train_label:index(1,ids[{{1,math.floor(train_nrow*rate_valid)}}]:type("torch.LongTensor"))

		vatrain_data[i] = train_data:index(1,ids[{{math.floor(train_nrow*rate_valid)+1,train_nrow}}]:type("torch.LongTensor"))
		vatrain_label[i] = train_label:index(1,ids[{{math.floor(train_nrow*rate_valid)+1,train_nrow}}]:type("torch.LongTensor"))
	end
end

collectgarbage();
