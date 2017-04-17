classdef WiSARD < handle
   properties
      discriminators
      classes
      nmemories
      nbits
      bits_order
   end
   
   methods
       % Constructor
       function obj = WiSARD(classes, input_size, nbits, bits_order)
         if nargin < 3
            error('Not enough input arguments');
         end
         if nargin < 4
            bits_order = randperm(input_size);
         end
         obj.classes = classes;
         obj.nbits = nbits;
         obj.nmemories = ceil(input_size / nbits);

         for c = 1:length(classes)
            obj.discriminators{c} = zeros(obj.nmemories, 2^obj.nbits);
         end

         obj.bits_order = bits_order;
      end

      

      function train(obj, data, classes, shuffle)
         if nargin < 4 || shuffle 
             data = obj.shuffleData( data );
         end
         
         dataAddr = obj.bin2Addr( data );
          
         for i = 1:size(dataAddr, 1)
            disc_idx = strcmp(obj.classes, classes{i});
            idxs = dataAddr(i, :) * obj.nmemories + [1:obj.nmemories]; 
            obj.discriminators{disc_idx}(idxs) = obj.discriminators{disc_idx}(idxs) + 1;
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
                %results(i, d) = sum( obj.discriminators{d}(idxs) .* (data(i, :) > 0) );
                results(i, d) = sum( obj.discriminators{d}(idxs) );
            end
            [~, idx] = max(results(i, :));
            y{i} = obj.classes{idx};
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
      
      function bleach(obj, method, threshold)
          if nargin < 2
              method = 'logarithm';
          end
         
          if strcmp(method, 'logarithm')
              for d = 1:length(obj.discriminators)
                  obj.discriminators{d}(:) = log( obj.discriminators{d}(:) + 1 );
              end
          elseif strcmp(method, 'threshold')
               allmemories = obj.getMergedDiscriminators();
               %max(allmemories(:))
               threshold = threshold * max(allmemories(:));
               for d = 1:length(obj.discriminators)
                   vals = obj.discriminators{d}(:);
                   vals( vals > threshold ) = vals( vals > threshold ) - threshold;
                   obj.discriminators{d}(:) = vals;
                   %obj.discriminators{d}(:) =  obj.discriminators{d}(:) / max(allmemories(:));
                   
              end
          elseif strcmp(method, 'discretize')
               for d = 1:length(obj.discriminators)
                   obj.discriminators{d}(:) = discretize(obj.discriminators{d}(:), 4);
                   %obj.discriminators{d}(:) =  obj.discriminators{d}(:) / max(allmemories(:));
                   
              end
          end
      end
      
      
      function [alldiscriminators] = getMergedDiscriminators(obj)
          alldiscriminators = [];
          for d=1:length(obj.discriminators)
              alldiscriminators = cat(1, alldiscriminators, obj.discriminators{d});
          end
      end
   end
end