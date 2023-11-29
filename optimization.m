classdef optimization < PROBLEM
% <multi/many> <real> 
% Benchmark MOP proposed by Deb, Thiele, Laumanns, and Zitzler

%------------------------------- Reference --------------------------------
% K. Deb, L. Thiele, M. Laumanns, and E. Zitzler, Scalable test problems
% for evolutionary multiobjective optimization, Evolutionary multiobjective
% Optimization. Theoretical Advances and Applications, 2005, 105-145.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        %% Default settings of the problem
        function Setting(obj)
            if isempty(obj.M); obj.M = 2; end
            if isempty(obj.D); obj.D = 6; end
            obj.lower    = zeros(1,obj.D);%下界
            obj.upper    = ones(1,obj.D);%上界
            obj.encoding = ones(1,obj.D);
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            [net1,net2] = obj.ParameterSet();
            % 将决策变量矩阵拆分为x1、x2和x3
            x1 = PopDec(:,1:2)';
            x2 = PopDec(:,3:4)';
            x3 = PopDec(:,5:6)';
            PopObj(:,1) = -net1([x2;x3])';
            PopObj(:,2) = -net2([x1;x2])';
        end
    end
end