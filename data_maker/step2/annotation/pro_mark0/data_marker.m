      function varargout = data_marker(varargin)
%DATA_MARKER M-file for data_marker.fig
%      DATA_MARKER, by itself, creates a new DATA_MARKER or raises the existing
%      singleton*.
%
%      H = DATA_MARKER returns the handle to a new DATA_MARKER or the handle to
%      the existing singleton*.
%
%      DATA_MARKER('Property','Value',...) creates a new DATA_MARKER using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to data_marker_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      DATA_MARKER('CALLBACK') and DATA_MARKER('CALLBACK',hObject,...) call the
%      local function named CALLBACK in DATA_MARKER.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help data_marker

% Last Modified by GUIDE v2.5 03-Aug-2015 20:23:13

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @data_marker_OpeningFcn, ...
    'gui_OutputFcn',  @data_marker_OutputFcn, ...
    'gui_LayoutFcn',  [], ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
end

% --- Executes just before data_marker is made visible.
% --- Author �� zhangyingying
function data_marker_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for data_marker
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes data_marker wait for user response (see UIRESUME)
% uiwait(handles.figure1);
end

% --- Outputs from this function are returned to the command line.
function varargout = data_marker_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
global start;
global con_save;
con_save = 0;
start = 0;
default_im = imread('default.jpg');

imshow(default_im);
varargout{1} = handles.output;
end

%% --- Executes on button press in openfile.
function openfile_Callback(hObject, eventdata, handles)
% hObject    handle to openfile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global label_loc;
global pop_number;
global img_num;
global current_pt;
global i_name;
global im_need;
global ponit_num;
global control;
global start;
if (start == 1)
    h = msgbox('You can only start one time!','Warning');
else
start = 1;
current_pt = 1;
ponit_num = 0;
control = 0;
%path = 'C:\Users\zyy16_000\Desktop\marking\immm\';
f_pa = mfilename('fullpath');
path_length = length(f_pa);
f_pa(path_length-20:path_length) = [];
path = f_pa;
path_1 = fullfile(path,'Image_data');

folder_list = dir(path_1);
folder_list = folder_list(3:end);
if(strcmpi(folder_list(1).name,'Thumbs.db'))
    folder_list(1) =[];
end
img_total = size(folder_list,1);
im_need = 0;
 med_1 = '.jpg';
 med_2 = '.jpeg';
 med_3 = '.png';
for i = 1:img_total;
    intermedia =  folder_list(i).name;
    str_length = size(intermedia,2);
    med_1_id = intermedia(str_length-3:str_length);
    med_2_id = intermedia(str_length-4:str_length);
    med_3_id = intermedia(str_length-3:str_length);
    if (strcmpi(med_1,med_1_id)...
            ||strcmpi(med_2,med_2_id)...
            ||strcmpi(med_3,med_3_id))
        im_need = im_need+1;
    end
end
if(im_need ==0)
    h = msgbox('You have finished annotated,there is no image data!!                                          Please add images to folder','Error','error');
else
    img_num = 1;
    label_loc = [];
    pop_number = 0;
    i_name = cell(1,img_total);
    
    for i_n = 1:img_total
        i_name{i_n} = folder_list(i_n).name;
    end
    im_ord = 1;  
    imglist = fullfile(path_1,folder_list(im_ord).name);
    label_image = imread(imglist);
    imshow(label_image);
    hold on;
    mark_crowd();
end
end
end



%% --- Executes on button press in push_cancel.
% function push_cancel_Callback(hObject, eventdata, handles)
% % hObject    handle to push_cancel (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% % global can_con;
% % can_con = 1;
% global len_loc;
% global label_loc;
% global h_plot;
% global current_pt;
% global pop_number;
%
% current_pt = current_pt-1;
% set(h_plot(current_pt),'Visible','off');
%  len_loc = length(label_loc);
%  label_loc(len_loc-1:len_loc) = [];
%  pop_number = length(label_loc)/2;
% end
%%  now cancel function
function push_cancel_Callback(hObject, eventdata, handles)
global start;
if(start == 1)
zoom off
cancel_point();
else
    h = msgbox('Please click start first!','Kindly Reminder');
end
end
%% --- Executes on button press in push_save.
function push_save_Callback(hObject, eventdata, handles)
% hObject    handle to push_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global label_loc;
global img_num;
global i_name;
global ponit_num;
global con_save;
global start;
if(start == 1)
%global im_need;
%global h_plot;
%global current_pt;
%global pop_number;

%name = sprintf('image_%d.mat',ann_num);
%img_name = sprintf('GT_image_%d.bmp',ann_num);
if (con_save ==1)
    h = msgbox('You have saved current iamge annotation, Please click next to get a new image','Warning');
else
if(isempty(label_loc))
    h = msgbox('You have not annotated data!!','Warning');
else
iden_letter = i_name{1}(1);
letter_ord = 1;
GT_name = '';
while(iden_letter~='.')
    GT_name = strcat(GT_name,i_name{img_num}(letter_ord));
    letter_ord = letter_ord+1;
    iden_letter = i_name{img_num}(letter_ord);
end
name = strcat('GT_',GT_name,'.mat');
img_name = strcat('GT_',GT_name,'.bmp');
lab_length = length(label_loc)/2;
label_ad = label_loc(1,1:2);
for i_la = 2:lab_length
    label_ad = [label_ad;label_loc(1,2*i_la-1:2*i_la)];
end
image_info{1}.location = label_ad;
image_info{1}.number = length(label_loc)/2;

save(name,'image_info');

pix=getframe(handles.axes1);
imwrite(pix.cdata,img_name);

f_pa = mfilename('fullpath');
path_length = length(f_pa);
f_pa(path_length-20:path_length) = [];
path = f_pa;
path_2 = fullfile(path,'GT_data');
path_mat = fullfile(path,'GT_mat');
movefile(img_name,path_2);
movefile(name,path_mat);

f_pa = mfilename('fullpath');
path_length = length(f_pa);
f_pa(path_length-20:path_length) = [];
path = f_pa;
path_3 = fullfile(path,'Image_data');
folder_list = dir(path_3);
folder_list = folder_list(3:end);
if(strcmpi(folder_list(1).name,'Thumbs.db'))
    folder_list(1) =[];
end
imglist = fullfile(path_3,folder_list(1).name);
path_4 = fullfile(path,'Annotated_data');
movefile(imglist,path_4);
ponit_num = ponit_num+length(label_loc)/2;
s_img = num2str(img_num);
s_pt = num2str(ponit_num);
set(handles.pt_num,'string',s_pt);
set(handles.ann_num,'string',s_img);
h = msgbox('Save successfully!','Save');
 con_save = 1;

% current_pt = 1;
% pop_number = 0;
% label_loc = [];
% h_plot = [];
% ann_num = ann_num+1;
%
% if(ann_num<=im_need)
% hold off;
% path = 'C:\Users\zyy16_000\Desktop\marking\immm\';
% folder_list = dir(path);
% folder_list = folder_list(3:end);
% imglist = fullfile(path,folder_list(ann_num).name);
% label_image = imread(imglist);
% imshow(label_image);
% hold on;
% mark_crowd();
% else
%     h = msgbox('Congratulations,you have finish annotating!','Success');
% end
end
end
else
    h = msgbox('Please click start first!','Kindly Reminder');
end
end


%% --- Executes on button press in zoom_in.
function zoom_in_Callback(hObject, eventdata, handles)
% hObject    handle to zoom_in (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global start;
if(start == 1)
h = zoom;
setAxesZoomMotion(h,handles.axes1,'both');
zoom on
set(h,'direction','in');
else
    h = msgbox('Please click start first!','Kindly Reminder');
end
end

% --- Executes on button press in zoom_out.
function zoom_out_Callback(hObject, eventdata, handles)
% hObject    handle to zoom_out (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global start;
if(start == 1)
h = zoom;
setAxesZoomMotion(h,handles.axes1,'both');
zoom on
set(h,'direction','out');
else
    h = msgbox('Please click start first!','Kindly Reminder');
end
end


%% --- Executes on button press in push_conti.
function push_conti_Callback(hObject, eventdata, handles)
% hObject    handle to push_conti (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global start;
if(start == 1)
zoom off
mark_crowd();
else
    h = msgbox('Please click start first!','Kindly Reminder');
end
end
%%      get the information from image
function mark_crowd()

global label_loc;
global h_plot;
global pop_number;
global current_pt;

set(gcf,'WindowButtonDownFcn',@MouseClickFcn);

pop_number = length(label_loc)/2;

    function MouseClickFcn(src,event)
        pt=get(gca,'CurrentPoint');
        x=pt(1,1);
        y=pt(1,2);
        h_plot(current_pt) = plot(x,y,'+', 'MarkerSize', 5, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'g');
        label_loc = [label_loc,x,y];
        current_pt = current_pt+1;
        
        
    end
end
function cancel_point()
global label_loc;
global h_plot;
global pop_number;
global current_pt;


set(gcf,'WindowButtonDownFcn',@MouseClickFcn);

pop_number = length(label_loc)/2;

    function MouseClickFcn(src,event)
        pt=get(gca,'CurrentPoint');
        x=pt(1,1);
        y=pt(1,2);
        if(pop_number>0)
            for i = 1:pop_number
                distance_id(i) = (x-label_loc(2*i-1))^2+(y-label_loc(2*i))^2;
            end
            [~,cancel_pt] = min(distance_id);
            
            current_pt = current_pt-1;
            set(h_plot(cancel_pt),'Visible','off');
            h_plot(cancel_pt) = [];
            label_loc(2*cancel_pt-1:2*cancel_pt) = [];
            pop_number = length(label_loc)/2;
        else
            h = msgbox('There is no point!!','Warning');
        end
    end
end

% --- Executes on button press in next.
function next_Callback(hObject, eventdata, handles)
% hObject    handle to next (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global label_loc;
global h_plot;
global current_pt;
global pop_number;
global img_num;
global im_need;
global start;
global con_save;
if(start == 1)
    if (con_save == 0)
    h = msgbox('You have not saved current annotation.If you have finished current image,Please click save.','Warning','error');
else
current_pt = 1;
pop_number = 0;
label_loc = [];
h_plot = [];
img_num = img_num+1;
if(img_num<=im_need)
    hold off;
  f_pa = mfilename('fullpath');
path_length = length(f_pa);
f_pa(path_length-20:path_length) = [];
path = f_pa;
path_5 = fullfile(path,'Image_data');
    folder_list = dir(path_5);
    folder_list = folder_list(3:end);
    if(strcmpi(folder_list(1).name,'Thumbs.db'))
    folder_list(1) =[];
    end
    imglist = fullfile(path_5,folder_list(1).name);
    label_image = imread(imglist);
    imshow(label_image);
    con_save = 0;
    hold on;
    mark_crowd();
    
else
    h = msgbox('Congratulations,you have finish annotating! If you want to annotate more image data, please add data to fold and reset this program','Success');
end
end
else
    h = msgbox('Please click start first!','Kindly Reminder');
end
end

%%


% --- Executes during object creation, after setting all properties.
function pt_num_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pt_num (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end

% --- Executes during object creation, after setting all properties.
function ann_num_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ann_num (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
