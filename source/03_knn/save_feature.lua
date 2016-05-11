
function save_feature(id, pred, path, feature_names)

	local fp = io.open(path, "w")
	
	local headwrite = table.concat(feature_names, ",")
	fp:write(headwrite.."\n")

	for i=1,#id do

		if i%100 == 0 then xlua.progress(i, #id) end

		local row = {id[i]}
		for k = 1,pred:size(2) do
			table.insert(row, pred[{i,k}])
		end
		local rowwrite = table.concat(row, ",")
		fp:write(rowwrite.."\n")

		if i%100 == 0 then collectgarbage() end
	end
	xlua.progress(#id, #id)

	fp:close()
end
