
------------------------------
-- function
------------------------------

function get_mean_std(data)

	local mean_ = torch.Tensor(data:size(2))
	local std_ = torch.Tensor(data:size(2))

	for i=1,data:size(2) do
		v = data[{{},i}]
		rm_na = v[torch.eq(v,v)]
		mean_[i] = rm_na:mean()
		std_[i] = rm_na:std()
	end

	std_:add(1e-20)

	return mean_, std_
end

function normalize_col(data, mean_, std_)

	for i = 1,data:size(2) do
		data[{{},i}]:add(-mean_[i])
		data[{{},i}]:div(std_[i])
	end

	return data
end
