close all; clear all; clc;

samplingrate = 500;  %original sampling rate = 1000Hz
% patient = ['patient001';'patient002';'patient003';'patient004';'patient005';
%     'patient006';'patient007';'patient008';'patient009';'patient010';
%     'patient011';'patient002';'patient003';'patient004';'patient005';
%     'patient001';'patient002';'patient003';'patient004';'patient005';
%     'patient001';'patient002';'patient003';'patient004';'patient005';
%     'patient001';'patient002';'patient003';'patient004';'patient005';
%     'patient001';'patient002';'patient003';'patient004';'patient005';
%     'patient001';'patient002';'patient003';'patient004';'patient005';
%     'patient001';'patient002';'patient003';'patient004';'patient005';
%     'patient001';'patient002';'patient003';'patient004';'patient005';];
patient = ['patient001';
'patient002';
'patient003';
'patient004';
'patient005';
'patient006';
'patient007';
'patient008';
'patient009';
'patient010';
'patient011';
'patient012';
'patient013';
'patient014';
'patient015';
'patient016';
'patient017';
'patient018';
'patient019';
'patient020';
'patient021';
'patient022';
'patient023';
'patient024';
'patient025';
'patient026';
'patient027';
'patient028';
'patient029';
'patient030';
'patient031';
'patient032';
'patient033';
'patient034';
'patient035';
'patient036';
'patient037';
'patient038';
'patient039';
'patient040';
'patient041';
'patient042';
'patient043';
'patient044';
'patient045';
'patient046';
'patient047';
'patient048';
'patient049';
'patient050';
'patient051';
'patient052';
'patient053';
'patient054';
'patient055';
'patient056';
'patient057';
'patient058';
'patient059';
'patient060';
'patient061';
'patient062';
'patient063';
'patient064';
'patient065';
'patient066';
'patient067';
'patient068';
'patient069';
'patient070';
'patient071';
'patient072';
'patient073';
'patient074';
'patient075';
'patient076';
'patient077';
'patient078';
'patient079';
'patient080';
'patient081';
'patient082';
'patient083';
'patient084';
'patient085';
'patient086';
'patient087';
'patient088';
'patient089';
'patient090';
'patient091';
'patient092';
'patient093';
'patient094';
'patient095';
'patient096';
'patient097';
'patient098';
'patient099';
'patient100';
'patient101';
'patient102';
'patient103';
'patient104';
'patient105';
'patient106';
'patient107';
'patient108';
'patient109';
'patient110';
'patient111';
'patient112';
'patient113';
'patient114';
'patient115';
'patient116';
'patient117';
'patient118';
'patient119';
'patient120';
'patient121';
'patient122';
'patient123';
'patient125';
'patient126';
'patient127';
'patient128';
'patient129';
'patient130';
'patient131';
'patient133';
'patient135';
'patient136';
'patient137';
'patient138';
'patient139';
'patient140';
'patient141';
'patient142';
'patient143';
'patient144';
'patient145';
'patient146';
'patient147';
'patient148';
'patient149';
'patient150';
'patient151';
'patient152';
'patient153';
'patient154';
'patient155';
'patient156';
'patient157';
'patient158';
'patient159';
'patient160';
'patient162';
'patient163';
'patient164';
'patient165';
'patient166';
'patient167';
'patient168';
'patient169';
'patient170';
'patient171';
'patient172';
'patient173';
'patient174';
'patient175';
'patient176';
'patient177';
'patient178';
'patient179';
'patient180';
'patient181';
'patient182';
'patient183';
'patient184';
'patient185';
'patient186';
'patient187';
'patient188';
'patient189';
'patient190';
'patient191';
'patient192';
'patient193';
'patient194';
'patient195';
'patient196';
'patient197';
'patient198';
'patient199';
'patient200';
'patient201';
'patient202';
'patient203';
'patient204';
'patient205';
'patient206';
'patient207';
'patient208';
'patient209';
'patient210';
'patient211';
'patient212';
'patient213';
'patient214';
'patient215';
'patient216';
'patient217';
'patient218';
'patient219';
'patient220';
'patient221';
'patient222';
'patient223';
'patient224';
'patient225';
'patient226';
'patient227';
'patient228';
'patient229';
'patient230';
'patient231';
'patient232';
'patient233';
'patient234';
'patient235';
'patient236';
'patient237';
'patient238';
'patient239';
'patient240';
'patient241';
'patient242';
'patient243';
'patient244';
'patient245';
'patient246';
'patient247';
'patient248';
'patient249';
'patient250';
'patient251';
'patient252';
'patient253';
'patient254';
'patient255';
'patient256';
'patient257';
'patient258';
'patient259';
'patient260';
'patient261';
'patient262';
'patient263';
'patient264';
'patient265';
'patient266';
'patient267';
'patient268';
'patient269';
'patient270';
'patient271';
'patient272';
'patient273';
'patient274';
'patient275';
'patient276';
'patient277';
'patient278';
'patient279';
'patient280';
'patient281';
'patient282';
'patient283';
'patient284';
'patient285';
'patient286';
'patient287';
'patient288';
'patient289';
'patient290';
'patient291';
'patient292';
'patient293';
'patient294';];

target = ['II';'I3';'VR';'VL';'VF';'V1';'V2';'V3';'V4';'V5';'V6';'Vx';'Vy';'Vz'];
Gain = 2000;
%normalizesize = 300;
windowsize = 150;
times = 1;
dim_input = windowsize/times;
% L = 50000;
test_size = 15000;

[P,~] = size(patient);
% [W,~] = size(DataBase);

for patientNum = 1:P
    inputpath = ['./ECG_Data/', deblank(patient(patientNum,:)),'/'];
    Dir_Info = dir(inputpath);
    DataBase = [];
    count = 0;
    for j = 1:length(Dir_Info)
        checkFile = strfind(Dir_Info(j).name ,'.mat');
        TF = isempty(checkFile);
        if  Dir_Info(j).isdir == 0 && TF == false && length(Dir_Info(j).name) < 15
            newStr = erase(Dir_Info(j).name ,'.mat');
            count = count + 1;
            newStr = strcat("'", newStr); 
            newStr = strcat(newStr, "'; ");
            %FileStr = [FileStr newStr];
            DataBase = [DataBase newStr];
        end
    end
    
    [~,W] = size(DataBase);
    for DataNumber = 1:W
       ECG_data = deblank(DataBase(DataNumber));
       ECG_data = extractBetween(ECG_data,"'","';");
       file = append(inputpath,ECG_data);
       [signal]= plotATM(file);
 
       [leadsNum,sigLength] = size(signal);
       out = zeros(leadsNum,sigLength); % out = zeros(15,115200);
       leads = zeros(leadsNum,floor(sigLength/2));  % leads = zeros(15,57600); ¦]¬°resample 1000->500

       d1 = designfilt('bandpassiir','FilterOrder',4, ...
            'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',150,'SampleRate',1000,'DesignMethod','butter');
       for i =1:15
           out(i,:) = filtfilt(d1,signal(i,:));
           leads(i,:) = resample(out(i,:),samplingrate,1000);
       end
       M = leads(1,:);  % M is lead I
       [~,R_peak,~] = pan_tompkin(M,samplingrate,0);
       %figure, 
       %plot(leads(1,:));

       for j = 1:length(R_peak)
           range = ceil(0.03*samplingrate);
           peakPeriod = R_peak(j)-range:R_peak(j)+range;
           peakPeriod(peakPeriod<=0) = [];
           peakPeriod(peakPeriod>length(M)) = [];
           peakSegment = M(peakPeriod);
           [~,idx] = max(peakSegment-min(M));
           R_peak(j) = peakPeriod(idx);
       end

       lead = leads(:,R_peak(3):R_peak(end-1)); %only lead I
       Lead_I = lead(1,:);
       [~,L] = size(lead); 
       input = zeros(L,times,dim_input);
       output = zeros(L,1);
    %    testinput = zeros(test_size,times,dim_input);
    %    testoutput = zeros(test_size,1);

       for t = 1:14 
           if t == 1
              Lead_target = lead(2,:);  %lead II
           elseif t == 2
              Lead_target = lead(3,:);
           elseif t == 3
              Lead_target = lead(4,:);
           elseif t == 4
              Lead_target = lead(5,:);
           elseif t == 5
              Lead_target = lead(6,:);
           elseif t == 6
              Lead_target = lead(7,:);
           elseif t == 7
              Lead_target = lead(8,:);
           elseif t == 8
              Lead_target = lead(9,:);
           elseif t == 9
              Lead_target = lead(10,:); 
           elseif t == 10
              Lead_target = lead(11,:);
           elseif t == 11
              Lead_target = lead(12,:);
           elseif t == 12
              Lead_target = lead(13,:);
           elseif t == 13
              Lead_target = lead(14,:);          
           else
              Lead_target = lead(15,:);  %lead Vz
           end
           %training data 
           for i = 1 : L-windowsize
                for j=1:times
                    input(i,j,1:dim_input) = Lead_I(i+int32(windowsize/times)*(j-1):i+int32(windowsize/times)*(j-1)+dim_input-1);

                end
                output(i,1) = Lead_target(windowsize-1+i); % original lead II or lead V2
           end
    %        %testing data 
    %        for i = 1+L : test_size+L
    %             for j=1:times
    %                 testinput(i-L,j,1:dim_input) = Lead_I(i+int32(windowsize/times)*(j-1):i+int32(windowsize/times)*(j-1)+dim_input-1);
    %             end
    %             testoutput(i-L,1) = Lead_target(windowsize-1+i);
    %        end  
           save([inputpath,num2str(ECG_data),'_',num2str(target(t,:)),'_I_RNN.mat'],'input','output'...
                    ,'L','dim_input','windowsize','times');
       end
    end
end