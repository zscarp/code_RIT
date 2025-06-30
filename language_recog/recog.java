import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class lab2 {
    public static class dect implements Serializable{
        private node root;
        private int MAXdepth = 3;
    }
    public static class node implements Serializable{
        private double gain = 0;
        private node lchild = null;
        private node rchild = null;
        private int att = -1;
        private int[] tag = {0,0,0,0,0};
        private boolean branch = true;
        private int depth = 0;
        private ArrayList<example> rec = new ArrayList<>();
        private int enc = 0;
        private int nlc = 0;
        private String predict;
        public String toString(){
            return ("total: "+this.rec.size()+" attribute:  "+this.att+" which is "
                    +this.branch+" en: "+this.enc+" nl: "+this.nlc+" informain gain: "+this.gain+" predict: "+this.predict);
        }
    }
    public static class example implements Serializable{
        private boolean[] att = new boolean[5];
        private int lang = 0;
        private double weight = 0;
        public example(boolean[] bool ,int lang){
            for (int i = 0; i < 5; i++)
                this.att[i] = bool[i];
            this.lang = lang;
        }
        public String toString(){
            String st = "";
            if (lang == 1)
                st = "nl";
            else if (lang == -1)
                st = "en";
            else
                System.out.println("check lang");
            return st+" "+att[0]+" "+att[1]+" "+att[2]+" "+att[3]+" "+att[4];
        }
    }

    public static class alg implements Serializable {
        private String[] words = new String[15];
        private  int lang = 0;
        private ArrayList<example> data = new ArrayList<>();
        private dect detree = new dect();
        private ArrayList<Double> weights = new ArrayList<>();
        private ArrayList<Integer> hypo = new ArrayList<>();
        private boolean[] classify = new boolean[5];
        private double[] err  = new double[5];
        public void read(String f, String method) throws FileNotFoundException {
            Scanner sc = new Scanner(new File(f));
            while (sc.hasNextLine()){
                String st = sc.next();
                int tag = 1;
                if (st.contains("nl|")){
                    this.lang = 1;
                    words[0] = st.substring(3).replaceAll("[,.?;:!()-]","");
                }

                else if (st.contains("en|")){
                    this.lang = -1;
                    words[0] = st.substring(3).replaceAll("[,.?;:!()-]","");
                }

                else {
                    tag = 0;
                    this.words[0] = st.replaceAll("[,.?;:!()-]","");
                    //System.out.print("input words: "+this.words[0]+" ");
                    //System.out.println("check scanner"+st);
                    //System.exit(-1);
                }

                for (int i = 1; i < 15; i++){
                    this.words[i] = sc.next().replaceAll("[,.?;:!()-]","");
                    //if (tag == 0)
                    //    System.out.print(this.words[i]+" ");
                }

                /*for (int j = 0; j < 15; j++)
                    System.out.print(words[j]+" ");
                System.out.print("\n");*/
                this.process(this.words,tag,this.lang, method);
            }
            this.detree.root = new node();
            for (int i = 0; i < this.data.size(); i++){
                this.detree.root.rec.add(this.data.get(i));
                if (this.data.get(i).lang == 1)
                    this.detree.root.nlc++;
                else
                    this.detree.root.enc++;
            }
        }

        public void process(String[] wd,int tag,int lang, String method){
            int l = wd.length;
            boolean[] bool = new boolean[5];
            int tg[] = {0,0,0,0,0};
            for (int i = 0; i < l; i++){
                if (wd[i].equals("de")|wd[i].equals("De"))
                    tg[0] = 1;
                if (wd[i].equals("a")|wd[i].equals("A")|wd[i].equals("an")|wd[i].equals("An"))
                    tg[1] = 1;
                if ((wd[i].length()>1)&&(wd[i].substring((wd[i].length())-2).equals("ed")))
                    tg[2] = 1;
                if (wd[i].contains("aa"))
                    tg[3] = 1;
                if (wd[i].contains("the")|wd[i].contains("The"))
                    tg[4] = 1;
            }
            for (int t = 0; t < 5; t++)
                bool[t] = tg[t] == 1;
            if (tag == 0 && method.equals("dt")){
                node tmp = this.detree.root;
                for (int i = 0; i < this.detree.MAXdepth; i++){
                    if (tmp.rchild == null && tmp.lchild == null){
                        //System.out.print("\n"+" predict: "+tmp.predict+"\n");
                        System.out.println( tmp.predict);
                        return;
                    }
                    else if (bool[i])
                        tmp = tmp.lchild;
                    else
                        tmp = tmp.rchild;
                }
                System.out.println(tmp.predict);
                return;
            }
            else if (tag == 0 && method.equals("ada")){
                double sign = 0;
                for (int i = 0; i < this.weights.size(); i++){
                    if (bool[this.hypo.get(i)])
                        sign += this.weights.get(i);
                    else
                        sign -= this.weights.get(i);
                }
                if (sign > 0)
                    System.out.println("nl");
                else if (sign < 0)
                    System.out.println("en");
                else
                    System.out.println("en/nl");

            }
            example ex = new example(bool,lang);
            this.data.add(ex);
        }

        public void adab(ArrayList<example> examples, int k){
            for (int i = 0; i < 5; i++)
                this.classify(examples, i);
            for ( int j = 0; j < examples.size(); j++ )
                examples.get(j).weight = (double) 1/examples.size();
            for (int p = 0; p < k; p++){
                int hypo = this.select();
                for (int t = 0; t < examples.size(); t++){
                    if (examples.get(t).att[hypo] == this.classify[hypo]){
                        if (examples.get(t).lang != 1)
                            this.err[hypo] += examples.get(t).weight;
                    }
                    else{
                        if (examples.get(t).lang != -1)
                            this.err[hypo] += examples.get(t).weight;
                    }

                }
                for (int s = 0; s < examples.size(); s++){
                    if (examples.get(s).att[hypo] == this.classify[hypo]){
                        if (examples.get(s).lang == 1)
                            examples.get(s).weight *= (this.err[hypo]/(1-this.err[hypo]));
                    }
                    else{
                        if (examples.get(s).lang == -1)
                            examples.get(s).weight *= (this.err[hypo]/(1-this.err[hypo]));
                    }
                }
                double norm = 0;
                for (int l = 0; l < examples.size(); l++)
                    norm += examples.get(l).weight;
                for (int n = 0; n < examples.size(); n++)
                    examples.get(n).weight /= norm;
                this.weights.add(Math.log((1-this.err[hypo])/this.err[hypo]));
                this.hypo.add(hypo);
                System.out.println(p+"th boost, hypotheses: attribute "+hypo+", error rate: "
                        +this.err[hypo]+" weight: "+this.weights.get(this.weights.size()-1));
            }
        }

        private int select(){
            double err = 2;
            int select = 0;
            for (int i = 0; i < 5; i++){
                if (this.err[i] < err){
                    err = this.err[i];
                    select = i;
                }

            }
            return select;
        }

        private void classify(ArrayList<example> examples,int tag){
            double nlc = 0;
            double nltrue = 0;
            double entrue = 0;
            for (int i = 0; i < examples.size(); i++){
                if (examples.get(i).lang == 1){
                    nlc++;
                    if (examples.get(i).att[tag])
                        nltrue += 1;
                }
                if (examples.get(i).lang == -1 && examples.get(i).att[tag])
                    entrue ++;
            }
            double enc = examples.size()-nlc;
            double nlfalse = nlc - nltrue;
            double enfalse = enc - entrue;
            if (nltrue+enfalse >= nlfalse+entrue){
                this.err[tag] = (nlfalse + entrue)/examples.size();
                this.classify[tag] = true;
            }
            else{
                this.err[tag] = (nltrue + enfalse)/examples.size();
                this.classify[tag] = false;
            }
        }

        public void dtlearn(node nd){
            if (nd.depth == this.detree.MAXdepth)
                return;
            if (nd.enc*nd.nlc == 0)
                return;
            this.dteval(nd);
            this.dtlearn(nd.lchild);
            this.dtlearn(nd.rchild);
        }

        private void dteval(node nd){
            int num = 0;
            double gain = 0;
            for (int i = 0; i < 5; i++){
                if (nd.tag[i] == 0){
                    if (portion(nd.rec,nd.nlc,i) > gain){
                        num = i;
                        gain = portion(nd.rec,nd.nlc,i);
                    }
                }
            }
            node lnode = new node();
            node rnode = new node();
            lnode.depth = nd.depth+1;
            rnode.depth = nd.depth+1;

            for (int c = 0; c < nd.rec.size(); c ++){
                if (nd.rec.get(c).att[num]){
                    lnode.rec.add(nd.rec.get(c));
                    if (nd.rec.get(c).lang == 1)
                        lnode.nlc++;
                    else
                        lnode.enc++;
                }

                else{
                    rnode.rec.add(nd.rec.get(c));
                    if (nd.rec.get(c).lang == 1)
                        rnode.nlc++;
                    else
                        rnode.enc++;
                }

            }
            for (int t = 0; t < 5; t++){
                lnode.tag[t] = nd.tag[t];
                rnode.tag[t] = nd.tag[t];
            }

            lnode.tag[num] = 1;
            rnode.tag[num] = -1;
            lnode.att = num;
            rnode.att = num;
            lnode.branch = true;
            rnode.branch = false;
            if (lnode.enc == lnode.nlc)
                lnode.predict = "equal";
            else
                lnode.predict = (lnode.enc>lnode.nlc? "en" : "nl");

            if (rnode.enc == rnode.nlc)
                rnode.predict = "equal";
            else
                rnode.predict = (rnode.enc>rnode.nlc? "en" : "nl");
            if (lnode.enc + lnode.nlc == 0)
                lnode.predict = "no example";
            if (rnode.enc + rnode.nlc == 0)
                rnode.predict = "no example";

            nd.gain = gain;
            nd.lchild = lnode;
            nd.rchild = rnode;
        }

        private double portion(ArrayList<example> arr, int nlc,int att){
            double count = 0;
            double nl = 0;
            for (int i = 0; i < arr.size(); i++){
                if (arr.get(i).att[att]){
                    count++;
                    if (arr.get(i).lang == 1)
                        nl++;
                }
            }
            return this.gain(nl,count-nl,nlc,arr.size()-count);
        }

        private double B(double p){
            if (p == 0 || p ==1)
                return 0;
            return -(p*Math.log(p)/Math.log(2)+(1-p)*Math.log(1-p)/Math.log(2));
        }

        private double gain(double pos, double neg, double first, double second){
            double b = B(first/(first+second));
            double rt = B(pos/(pos+neg))*(pos+neg)/(first+second);
            double rf = B((first-pos)/(first+second-pos-neg))*(first+second-pos-neg)/(first+second);
            return b - rf - rt;
        }

        private void printree(node nd){
            String br = "";
            if (nd.depth == 0){
                System.out.println ("total: "+nd.rec.size()+" en: "+nd.enc+" nl: "+nd.nlc);
            }
            else {
                for (int i = 0; i < nd.tag.length; i++){
                    if (nd.tag[i] == 1)
                        br += "T";
                    else if (nd.tag[i] == -1)
                        br += "F";
                    else
                        br += "_";
                }
                System.out.print("branch "+br+" : ");
                System.out.println(nd);
            }
            if (nd.lchild != null)
                printree(nd.lchild);
            if (nd.rchild != null)
                printree(nd.rchild);
        }
    }




    public static void main(String[] args) throws IOException, ClassNotFoundException {
        if (args.length == 4){
            if (!args[0].equals("train")){
                System.out.println("check parameter");
                System.exit(-1);
            }

            alg a = new alg();
            a.read(args[1],args[3]);
            if (args[3].equals("dt")){
                a.dtlearn(a.detree.root);
                a.printree(a.detree.root);
                ObjectOutputStream ost = new ObjectOutputStream(new FileOutputStream(args[2]));
                ost.writeObject(a);
                ost.close();
            }
            else if (args[3].equals("ada")){
                a.adab(a.data,5);
                ObjectOutputStream ost = new ObjectOutputStream(new FileOutputStream(args[2]));
                ost.writeObject(a);
                ost.close();
            }

        }

        else if (args.length == 3){
            if (!args[0].equals("predict")){
                System.out.println("check parameter");
                System.exit(-1);
            }
            File file = new File(args[1]);
            ObjectInputStream ist = new ObjectInputStream(new FileInputStream(file));
            alg ina = (alg) ist.readObject();
            if (args[1].equals("decision_tree"))
                ina.read(args[2],"dt");
            else if (args[1].equals("adaboost"))
                ina.read(args[2],"ada");
            else
                System.out.println("check parameters");
            ist.close();
        }

    }
}
