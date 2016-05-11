
------------------------------
-- function
------------------------------

function auc(targets, pred)
	local neg = pred[torch.ne(targets,1)]
	local pos = pred[torch.eq(targets,1)]   
	if neg:nElement() == 0 or pos:nElement() == 0 then
		print('warning, there is only one class')
	end

	local C = 0
	for i=1,(#pos)[1] do
 		for j=1,(#neg)[1] do
 			if neg[j]<pos[i] then
 				C = C+1
 			elseif neg[j]==pos[i] then
            	C = C+0.5
         	end
		end
	end

	local AUC = C/((#neg)[1]*(#pos)[1])

	return AUC
end
