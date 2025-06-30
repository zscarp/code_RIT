/*
 * reference: dump_func from weijie Zhao 
 */
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <random>
#include <string.h>
#include <cstring>
#include <memory>
std::mt19937 generator(123456789);
std::normal_distribution<double> dist(0.0,1.0);
using namespace std;

struct sample{
	vector<double> feature;
	vector<int> label;
	int layer;
};

sample readl(char* line, int feature, int label, int len){
	sample ret;
	ret.feature.resize(feature);
	ret.label.resize(label);
	for (int i = 0; i < feature; i++){
		float tmp = 0;
      		sscanf(line,"%f",&tmp);
		cout<<tmp<<endl;
		ret.feature[i] = tmp;
		cout<<ret.feature[i]<<endl;
	}
	for (int i = 0; i < label; i++){
		float tmp = 0;
      		sscanf(line,"%f",&tmp);
		ret.label[i] = tmp;
	}
	return ret;
}

vector<sample> readata(FILE* fin, int row, int feature, int label){
	vector<sample> ret;
	ret.resize(row);
	/*char* buffer = (char*)malloc(sizeof(char*)*ftell(fin));
	vector<int> lc;
	lc.resize(row);
	int ls = 0;
	int r = fread(buffer, 1, ftell(fin),fin);
	for (int i = 0; i < ftell(fin); i++){
		if (buffer[i] == EOF){
			lc[ls] = i;
			break;
		}
		else{
			if(buffer[i] == '\n'){
				lc[ls] = i;
				ls++;
			}
		}
	}
	for (int i = 0; i < row; i++){
		char* line = &buffer[lc[i]];
		if (i == 0)
			ret[i] = readl(line, feature, label,lc[0]);
		else{
			ret[i] = readl(line, feature, label,lc[i]-lc[i-1]);
		
		}
	}*/
	for (int i = 0; i < row; i++){
		sample node;
		double tmp = 0;
		for (int j = 0; j < feature; j++){
			fscanf(fin, "%lf",&tmp);
			node.feature.push_back(tmp);
		}
		for (int j = 0; j < label; j++){
			fscanf(fin, "%lf",&tmp);
			node.label.push_back(tmp);
		}
		ret[i] = node;
	}

	return ret;
}
FILE* dump_func(int row, int feature, int label){
//	printf("dumping  ");
	std::string file_name = "sample.in";
	FILE* fp = fopen(file_name.c_str(),"w");
  std::vector<std::vector<double>> data;
  for(int i = 0;i < row;++i){
    std::vector<double> tmp;
    for(int j = 0;j < feature;++j)
      tmp.push_back(dist(generator));
    for(int j = 0;j < label;++j)
      tmp.push_back(rand()%2);
    data.push_back(tmp);
    fprintf(fp,"%f",tmp[0]);
    for(int j = 1;j < feature+label;++j)
      fprintf(fp," %f",tmp[j]);
    fprintf(fp,"\n");
  }
  fclose(fp);
  FILE* fin = fopen(file_name.c_str(),"r");
  return fin;
}

void dataproc(vector<sample> data ,int layers){
	double ts = data.size()*0.8;
	for (int i = 0; i < ts; i++){
		int l = 1;
		while(l < layers){
			if (i < l*ts/layers){
				break;
			}
			l++;
		}
		data[i].layer = l-1;
	}

}

vector<double> sgd(int idx, vector<sample> data, vector<sample> samples, vector<double> wht, int layers,int layer, int batch, double lr, int epoch, int feature, int lb){
	double ts = data.size()*0.8;
	int unit = ts*(layer+1)/layers-ts*layer/layers;
	vector<double> wt;
	//vector<double> gr;
	int fi = layer*feature/layers;
	int sc = (layer+1)*feature/layers;
	if (idx == 0){
		wt.resize(sc-fi);
		for (int i = 0; i < wt.size(); i++){
			wt[i] = 1;
		}
	}
	else{
		wt = wht;
	}
	vector<sample> bat;
	if (batch == 0){
		bat = samples;	
	}
	else{
		bat.resize(batch);
		int step = ts/(batch*(2+rand()%3));
		int ini = rand()%step;
		int loc = ts*layer/layers;
		for (int i = 0; i < batch; i++){
			loc += step;
			sample node;
			node.feature = data[loc].feature;
			node.label = data[loc].label;
			node.layer = data[loc].layer;
			bat[i] = node;
		}
	}
	vector<double> x;
	x.resize(sc-fi);
	double sum = 0;
	vector<double> gd;
	gd.resize(sc-fi);
	for (int i = 0; i < sc-fi; i++){
		sum = 0;
		gd[i] = 0;
		for (int j = 0; j < bat.size(); j++){
			sum += bat[j].feature[fi+i];
		}
		x[i] = sum;
	}
	for (int e = 0; e < epoch; e++){
		sum = 0;
#pragma omp paralell for num_threads(8)
		for (int j = 0; j < batch; j++){
			double prd = 0;
			for (int k = 0; k < sc-fi; k++){
				prd += bat[j].feature[fi+k]*wt[k];		
				//cout<<"pr: "<<prd<<endl;
			}	
			//if (prd > 0 )
			//	prd = 1;
			//else
			//	prd = 0;
			double cum = prd-bat[j].label[lb];
			//cout<<" cum: "<<cum<<endl;
			sum += cum;
		}
		for (int i = 0; i < sc-fi; i++){
			gd[i] += sum/x[i];
			wt[i] -= gd[i]*lr;
			//cout<<wt[i];
			//cout<<" gd: "<<gd[i];
		}
		//cout<<endl;
	}
	return wt;
}

double acc(vector<sample> data, vector<double> wt, int layers, int layer,int feature, int lb){
	double ts = data.size()*0.8;
	int unit = ts*(layer+1)/layers-ts*layer/layers;
	int fi = layer*feature/layers;
	int sc = (layer+1)*feature/layers;
	double count = 0;
	double sum = 0;
#pragma omp paralell for num_threads(8)
	for (int i = ts; i < data.size(); i++){
		count ++;
		double prd = 0;
		for (int j = 0; j< sc-fi; j++ ){
			prd += data[i].feature[fi+j]*wt[j];
		}
		if (prd > 0.5 )
			prd = 1;
		else
			prd = 0;
		sum += (prd == data[i].label[lb])? 1:0;
	}
	return sum/count;
	
}

vector<sample> errors (vector<sample> data, vector<double> wup, vector<double> wdn,int layers, int cur_layer, int feature,int lb ){
	vector<sample> ers;
	double ts = data.size()*0.8;
	int low = ts*(cur_layer-1)/layers;
	int unit = ts*(cur_layer)/layers-low;
	int fl = (cur_layer-1)*feature/layers;
	int fi = cur_layer*feature/layers;
	int sc = (cur_layer+1)*feature/layers;

#pragma omp paralell for num_threads(8)
	for(int i = 0; i < unit; i++){
		double pru = 0;
		double prd = 0;
		for (int j = 0; j < sc-fi; j++){
			pru += data[i+low].feature[fi+j]*wup[j];
			prd += data[i+low].feature[fl+j]*wdn[j];
		}
		if (prd > 0.5 )
			prd = 1;
		else
			prd = 0;
		if (pru > 0.5 )
			pru = 1;
		else
			pru = 0;
		if (prd != pru){
			sample node;
			node.feature = data[i+low].feature;
			node.label = data[i+low].label;
			ers.push_back(node);
		}
	}
	//cout<<"errors: "<<ers.size()<<endl;
	return ers;
}

vector<double> update(vector<sample> data, vector<sample> samples,vector<double> weight, int layers,int layer, double lr, int epoch, int feature, int lb){
	vector<double> wt = weight;
	vector<double> ret = sgd(1,data,samples,weight,layers,layer,0,lr,epoch,feature,lb);
	return ret;
}


int main(int argc, char* argv[]) {
  //time_t st = time(NULL);	
  //FILE* fin = fopen(argv[1],"r");
  if (argc!= 5){
  	printf("missing argument!");
	exit(-1);
  }

  FILE* fout = fopen(argv[4],"w");
  int row = atoi(argv[1]);
  int feature = atoi(argv[2]);
  int label = atoi(argv[3]);
  

  FILE* fin = dump_func(row, feature, label);
  vector<sample> data = readata(fin,row, feature, label);

  for (int i = 0; i < row; i++){
  	for (int j = 0; j < feature; j++){
		fprintf(fout, " %lf",data[i].feature[j]);
	}
  	for (int j = 0; j < label; j++){
		fprintf(fout, " %d",data[i].label[j]);
	}
	fprintf(fout,"\n");
  }
  
  dataproc(data,4);
  vector<vector<double>> wt;
  vector<double> tmp;
  wt.resize(label);
  cout<<"single layer: "<<endl;
  for (int i = 0; i < 4; i++){
  	wt[i] = sgd(0,data,data,tmp,4,i,40,0.01,150,feature,i);
  	cout<<"acc: "<<acc(data,wt[i],4,i,feature,i)<<endl;
  }
  cout<<"layerwise: "<<endl;

  for (int i=1; i < 4; i++){
	vector<sample> samples = errors(data,wt[i],wt[i-1],4,i,feature,i);
	wt[i] = update(data, samples, wt[i], 4,i, 0.02, 150, feature, i);
  	cout<<"acc: "<<acc(data,wt[i],4,i,feature,i)<<endl;
  }

  fclose(fin);
  fclose(fout);
}  

