close all; clear all; clc;

fid = fopen('PyCommand.txt','wt');
% DataBase = ['s0171lrem';'s0102lrem'; 's0130lrem'; 's0155lrem';'s0334lrem';'s0201_rem'; 's0144lrem';'s0297lrem';'s0256lrem';'s0299lrem';];
% DataBase = ['s0078lrem';'s0316lrem';'s0138lrem';'s0136lrem'];

patient = ['patient001';'patient002';'patient003';'patient004';'patient005';...
'patient006';'patient007';'patient008';'patient009';'patient010';'patient011';...
'patient012';'patient013';'patient014';'patient015';'patient016';'patient017';...
'patient018';'patient019';'patient020';'patient021';'patient022';'patient023';...
'patient024';'patient025';'patient026';'patient027';'patient028';'patient029';...
'patient030';'patient031';'patient032';'patient033';'patient034';'patient035';...
'patient036';'patient037';'patient038';'patient039';'patient040';'patient041';...
'patient042';'patient043';'patient044';'patient045';'patient046';'patient047';...
'patient048';'patient049';'patient050';'patient051';'patient052';'patient053';...
'patient054';'patient055';'patient056';'patient057';'patient058';'patient059';...
'patient060';'patient061';'patient062';'patient063';'patient064';'patient065';...
'patient066';'patient067';'patient068';'patient069';'patient070';'patient071';...
'patient072';'patient073';'patient074';'patient075';'patient076';'patient077';...
'patient078';'patient079';'patient080';'patient081';'patient082';'patient083';...
'patient084';'patient085';'patient086';'patient087';'patient088';'patient089';...
'patient090';'patient091';'patient092';'patient093';'patient094';'patient095';...
'patient096';'patient097';'patient098';'patient099';'patient100';'patient101';...
'patient102';'patient103';'patient104';'patient105';'patient106';'patient107';...
'patient108';'patient109';'patient110';'patient111';'patient112';'patient113';...
'patient114';'patient115';'patient116';'patient117';'patient118';'patient119';...
'patient120';'patient121';'patient122';'patient123';'patient125';'patient126';...
'patient127';'patient128';'patient129';'patient130';'patient131';'patient133';...
'patient135';...
% 'patient136';
'patient137';'patient138';'patient139';'patient140';...
'patient141';'patient142';'patient143';...
%'patient144';
'patient145';'patient146';...
'patient147';'patient148';'patient149';'patient150';'patient151';'patient152';...
'patient153';'patient154';'patient155';'patient156';'patient157';'patient158';...
'patient159';'patient160';'patient162';'patient163';'patient164';'patient165';...
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
%'patient285';
'patient286';
'patient287';
'patient288';
'patient289';
'patient290';
'patient291';
'patient292';
'patient293';
'patient294';];

[P,~] = size(patient);
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
            DataBase = [DataBase newStr];
        end
    end
    DataBase = convertCharsToStrings(DataBase);
    [~,W] = size(DataBase);
    for DataNumber = 1:W
       ECG_data = deblank(DataBase(DataNumber));
       ECG_data = extractBetween(ECG_data,"'","';");
       file = append(inputpath,ECG_data);
        mkdir([inputpath,num2str(ECG_data)]);
        filepath =[deblank(patient(patientNum,:)),'/',convertStringsToChars(ECG_data)];
        for j = 1:5
             text1 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                    filepath,' --ot _II'];
             text2 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                    filepath,' --ot _I3'];
             text3 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _VR'];
             text4 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                    filepath,' --ot _VL'];
             text5 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _VF'];
             text6 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _V1'];        
             text7 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _V2'];
             text8 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _V3'];
             text9 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _V4'];
             text10 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _V5'];
             text11 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _V6'];
             text12 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _Vx'];
             text13 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _Vy'];
             text14 = ['python ECG.py --train True --m RNN --n ',num2str(j),' --p ',...
                     filepath,' --ot _Vz'];
    %         disp(text1);
    %         disp(text2);
            fprintf(fid, '%s\n',text1);
            fprintf(fid, '%s\n',text2);
            fprintf(fid, '%s\n',text3);
            fprintf(fid, '%s\n',text4);
            fprintf(fid, '%s\n',text5);
            fprintf(fid, '%s\n',text6);
            fprintf(fid, '%s\n',text7);
            fprintf(fid, '%s\n',text8);
            fprintf(fid, '%s\n',text9);
            fprintf(fid, '%s\n',text10);
            fprintf(fid, '%s\n',text11);
            fprintf(fid, '%s\n',text12);
            fprintf(fid, '%s\n',text13);
            fprintf(fid, '%s\n',text14);
        end
    end
end
fclose(fid);
