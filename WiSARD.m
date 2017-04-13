classdef WiSARD < handle
   properties
      descriminators
      classes
      nmemories
      nbits
      bits_order
   end
   methods
      function obj = WiSARD(classes, input_size, nbits)
         if nargin < 3
            error('Not enough input arguments');
         end
         obj.classes = classes;
         obj.nbits = nbits;
         obj.nmemories = ceil(input_size / nbits);

         for c = 1:length(classes)
            obj.descriminators{c} = zeros(obj.nmemories, 2^obj.nbits);
         end

         obj.bits_order = randperm(input_size);
      end

      

      function train(obj, data, classes)        
         % todo check this!
         data = obj.bin2Addr( obj.shuffleData( data ) );
          
         for i = 1:size(data, 1)
            disc_idx = strcmp(obj.classes, classes{i});
            idxs = data(i, :) * obj.nmemories + [1:obj.nmemories]; 
            obj.descriminators{disc_idx}(idxs) = obj.descriminators{disc_idx}(idxs) + 1;
         end
         
      end


      function [data_shuffled] = shuffleData(obj, data)
         data_shuffled = data;
         for i = 1:size(data, 1)
            data_shuffled(i, :) = data(i, obj.bits_order);
         end
      end


      function [addrData] = bin2Addr(obj, data)
         addrData = zeros(size(data, 1), obj.nmemories);

         k = 1;
         squares = 2.^[obj.nbits-1:-1:0];
         for i = 1:obj.nbits:size(data, 2)-obj.nbits
            addrData(:,k) = double(data(:, i:i+obj.nbits-1)) * squares';
            k = k+1;
         end

      end
      
      
      function [y, results] = test(obj, data)
         % todo check this!
         data = obj.bin2Addr( obj.shuffleData( data ) );
         
         results = zeros( size(data, 1), length(obj.classes) );
         y = cell(1, size(data, 1));

         for i = 1:size(data, 1)
            for d = 1:length(obj.classes)
                idxs = data(i, :) * obj.nmemories + [1:obj.nmemories]; 
                results(i, d) = sum(obj.descriminators{d}(idxs));
            end
            [~, idx] = max(results(i, :));
            y{i} = obj.classes{idx};
         end

      end 
      
      
   end
end