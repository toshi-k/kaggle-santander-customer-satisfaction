
------------------------------
-- function
------------------------------

function normalize_row(data)

	for i = 1,data:size(1) do
		datai = data[{i}]
		rm_na = datai[torch.eq(datai,datai)]
		rm_na:pow(2)
		v = torch.sum(rm_na)
		data[{i}]:div(math.sqrt(v))
	end

	return data
end
