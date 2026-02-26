function T = GabrielBridge(data, attributes, varargin)
% GABRIELBRIDGE The bridge between Human and Gabriel (GABRIEL Toolkit Interface).
%
%   T = GABRIEL_RATE(data, attributes)
%       Default: rates ('rate') the input data based on the provided attributes.
%
%   T = GABRIEL_RATE(data, attributes, 'Task', task_type)
%       Executes a specific GABRIEL task. Supported tasks:
%       'rate', 'classify', 'extract', 'rank', 'codify', 'paraphrase',
%       'compare', 'discover', 'bucket', 'seed', 'ideate', 'deidentify'.
%
%   INPUTS:
%     data       - MATLAB table, cell array of strings, or [] (for seed/ideate).
%     attributes - Task configuration:
%                  - For 'rate'/'extract'/'rank': struct mapping names to definitions.
%                  - For 'classify'/'discover': struct mapping labels to definitions.
%                  - For 'paraphrase'/'seed'/'deidentify': string instructions or struct.
%
%   NAME-VALUE OPTIONS:
%     'Task'         - (string) GABRIEL task type. Default: 'rate'.
%     'ColumnName'   - (string) Target column name. Default: first column.
%     'CircleColumnName', 'SquareColumnName' - (string) Paired columns for
%                      'compare' or 'discover' tasks.
%     'ProfileName'  - (string) Config profile name (e.g., 'deepseek').
%     'SaveDir'      - (string) Cache directory. Default: './gabriel_results'.
%     'ConfigPath'   - (string) Path to api_config.json.
%     'NRuns'        - (integer) Number of runs/passes. Default: 1.
%     'ResetFiles'   - (logical) Ignore existing cache. Default: true.
%
%   EXAMPLE:
%     % Classification
%     labels = struct('Pos', 'Positive', 'Neg', 'Negative');
%     T = gabriel_rate({'Good'}, labels, 'Task', 'classify');

% 1. Parse Inputs
p = inputParser;
p.KeepUnmatched = true;
addRequired(p, 'data');
addRequired(p, 'attributes');
addParameter(p, 'Task', 'rate', @ischar);
addParameter(p, 'ColumnName', '', @ischar);
addParameter(p, 'CircleColumnName', '', @ischar);
addParameter(p, 'SquareColumnName', '', @ischar);
addParameter(p, 'SaveDir', './gabriel_results', @ischar);
addParameter(p, 'ConfigPath', fullfile(pwd, 'api_config.json'), @ischar);
addParameter(p, 'ProfileName', '', @ischar);
addParameter(p, 'NRuns', 1, @isnumeric);
addParameter(p, 'NParallels', [], @isnumeric);
addParameter(p, 'ResetFiles', true, @islogical);
parse(p, data, attributes, varargin{:});

taskType = lower(p.Results.Task);

% 2. Prepare Data
if isempty(data) && ismember(taskType, {'seed', 'ideate'})
    % seed/ideate don't necessarily need input data
    py_data = py.dict();
else
    % Convert cell array to table if needed
    if iscell(data)
        data = table(data(:), 'VariableNames', {'text_col'});
    end

    % Determine column name
    if isempty(p.Results.ColumnName) && ~isempty(data)
        colName = data.Properties.VariableNames{1};
    else
        colName = p.Results.ColumnName;
    end

    % Convert MATLAB table to Python dict
    args = {};
    for i = 1:width(data)
        name = data.Properties.VariableNames{i};
        val = data.(name);
        if isstring(val) || iscellstr(val)
            val = cellstr(val);
            val = val(:)'; % 1D row
        end
        args{end+1} = name; %#ok<AGROW>
        args{end+1} = val; %#ok<AGROW>
    end
    py_data = py.dict(pyargs(args{:}));
end

% Convert attributes to Python dict (or pass as string)
if ischar(attributes) || isstring(attributes)
    py_attrs = attributes;
elseif isstruct(attributes)
    attr_fields = fieldnames(attributes);
    attr_args = {};
    for i = 1:length(attr_fields)
        f = attr_fields{i};
        attr_args{end+1} = f; %#ok<AGROW>
        attr_args{end+1} = attributes.(f); %#ok<AGROW>
    end
    py_attrs = py.dict(pyargs(attr_args{:}));
else
    py_attrs = py.dict(attributes);
end

% 3. Python Environment Setup
P = py.sys.path;
if count(P, pwd) == 0, insert(P, int32(0), pwd); end

% 4. Call GABRIEL Wrapper
pyKwargsCell = {'reset_files', p.Results.ResetFiles, 'n_runs', int32(p.Results.NRuns)};
if ~isempty(colName), pyKwargsCell(end+1:end+2) = {'column_name', colName}; end
if ~isempty(p.Results.CircleColumnName), pyKwargsCell(end+1:end+2) = {'circle_column_name', p.Results.CircleColumnName}; end
if ~isempty(p.Results.SquareColumnName), pyKwargsCell(end+1:end+2) = {'square_column_name', p.Results.SquareColumnName}; end
if ~isempty(p.Results.NParallels), pyKwargsCell(end+1:end+2) = {'n_parallels', int32(p.Results.NParallels)}; end

% Add unmatched
unmatched = p.Unmatched;
fn = fieldnames(unmatched);
for i = 1:length(fn)
    pyKwargsCell(end+1:end+2) = {fn{i}, unmatched.(fn{i})}; %#ok<AGROW>
end

pyKwargs = pyargs(pyKwargsCell{:});

try
    res = py.gabriel_wrapper.run_gabriel_task(...
        taskType, py_data, py_attrs, p.Results.SaveDir, ...
        p.Results.ConfigPath, p.Results.ProfileName, pyKwargs);

    % 5. Convert Results back to MATLAB
    if isa(res, 'py.dict')
        T = pyDictToTable(res);
    elseif isa(res, 'py.bool') || isa(res, 'py.int') || isa(res, 'py.float') || isa(res, 'py.str')
        T = res; % Return simple types directly
    else
        T = res;
    end

catch ME
    fprintf('Error in GABRIEL %s: %s\n', taskType, ME.message);
    rethrow(ME);
end
end

function T = pyDictToTable(py_dict)
% Helper to convert py.dict (from to_dict('list')) to MATLAB table
keys = cell(py.list(py_dict.keys()));
mat_keys = cellfun(@char, keys, 'UniformOutput', false);
T = table();
for i = 1:length(mat_keys)
    k = mat_keys{i};
    val = py_dict{k};
    if isa(val, 'py.dict')
        % Recurse for nested dicts (e.g. discover results)
        T.(k) = pyDictToTable(val);
    else
        % Convert list to cell or array
        list_val = cell(val);
        % Try to convert to double if it looks numeric
        try
            T.(k) = cellfun(@double, list_val)';
        catch
            T.(k) = cellfun(@(x) string(char(x)), list_val)';
        end
    end
end
end
