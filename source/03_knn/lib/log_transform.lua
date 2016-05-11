
------------------------------
-- function
------------------------------

function get_mins(data)

	local mins = torch.Tensor(data:size(2))

	for i = 1,data:size(2) do
		v = data[{{},i}]
		rm_na = v[torch.eq(v,v)]
		mins[i] = rm_na:min()
	end

	return mins
end

function log_transform(data, mins)

	for i = 1,data:size(2) do

		data[{{},i}]:add(-mins[{i}])
		local t = data[{{},i}]
		t[torch.lt(t,0)] = 0
		t:log1p()
		data[{{},i}] = t
	end

	return data
end
