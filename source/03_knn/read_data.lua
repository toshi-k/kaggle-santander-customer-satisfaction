
------------------------------
-- function
------------------------------

function table_count(tb)
	local count = 0
	for _ in pairs(tb) do count = count + 1 end
	return count
end

function read_data(path, istrain)
	local csv_data = csvigo.load{path = path}

	local id = csv_data["ID"]
	local data_nrow = #id

	print("data_nrow:" .. data_nrow)

	local label
	if istrain then
		label = torch.Tensor(csv_data["train_target"])
	end

	data_ncol = table_count(csv_data)
	print("data_ncol:" .. data_ncol)

	if istrain then
		data_ncol = data_ncol - 2
	else
		data_ncol = data_ncol - 1
	end

	local data = torch.Tensor(data_nrow, data_ncol)

	local count = 1
	for feature_name, ver in orderedPairs(csv_data) do
		-- print(feature_name)
		if feature_name ~= "train_target" and feature_name ~= "ID" then
			xlua.progress(count, data_ncol)
			-- print("feature_name: " .. feature_name .. " count:" .. count)

			for i,value in ipairs(ver) do

				if value == "NA" then
					data[{i,count}] = 'nan'
				else
					data[{i,count}] = tonumber(value)
				end
			end
			if count % 10 == 0 then collectgarbage() end

			count = count + 1
		end
	end
	xlua.progress(data_ncol, data_ncol)

	data = data:float()
	if istrain then
		label = label:int()
	end

	return data, label, id
end
