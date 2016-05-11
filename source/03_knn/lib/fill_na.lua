
------------------------------
-- function
------------------------------

function get_mean_(data)
	local mean_ = torch.Tensor(data:size(2))
	for i=1,data:size(2) do
		v = data[{{},i}]
		rm_na = v[torch.eq(v,v)]
		mean_[i] = rm_na:mean()
	end

	return mean_
end

function fill_na(data, mean_)

	for i=1,data:size(2) do
		local v = data[{{},i}]
		data[{{},i}][torch.ne(v,v)] = mean_[i]
	end

	return data
end
