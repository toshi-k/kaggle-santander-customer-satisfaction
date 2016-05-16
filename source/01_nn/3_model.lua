
------------------------------
-- library
------------------------------

require 'torch'
require 'image'
require 'nn'
require 'cunn'

------------------------------
-- function
------------------------------

-- input dimensions
nfeats = train_data:size(2)
print("nfeats: " .. tostring(nfeats))

function newmodel()

	-- model:
	local model = nn.Sequential()
	local hid1 = math.random(64,256)
	local hid2 = math.random(32,256)
	local drop1 = math.random()*0.5+0.1
	local drop2 = math.random()*0.5+0.1
	local slope1 = math.random()*0.3
	local slope2 = math.random()*0.3

	linear1 = nn.Linear(nfeats, hid1)
	model:add(linear1)

	model:add(nn.LeakyReLU(slope1))
	model:add(nn.Dropout(drop1))

	linear2 = nn.Linear(hid1, hid2)
	model:add(linear2)

	model:add(nn.LeakyReLU(slope2))
	model:add(nn.Dropout(drop2))

	linear3 = nn.Linear(hid2, 1)
	model:add(linear3)

	model:add(nn.Sigmoid())

	return model
end

------------------------------
-- main
------------------------------

model = newmodel()
print(model)

vamodels = {}
if opt.va > 0 then
	for i = 1,opt.va do
		vamodels[i] = model:clone()
	end
end

-- loss function
criterion = nn.BCECriterion()
print '==> here is the loss function:'
print(criterion)
