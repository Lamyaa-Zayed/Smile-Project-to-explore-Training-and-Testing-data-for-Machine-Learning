package smile.sm;

public class Passenger {
    static int id = 0;
    Integer passengerId;
    Integer pclass;
    Integer survived;
    String name;
    String sex;
    Double age;
    Integer sibSp;
    Integer parch;
    String ticket;
    Double fare;
    String cabin;
    String embarked;
    String homeDest;
    String boat;
    Integer body;
public Passenger(){

}

    public Passenger(Integer pclass, Integer survived, String name, String sex, Double age, Integer sibSp, Integer parch
            , String ticket, Double fare, String cabin, String embarked, String homeDest, String boat, Integer body) {
        this.pclass = pclass;
        this.survived = survived;
        this.name = name;
        this.sex = sex;
        this.age = age;
        this.sibSp = sibSp;
        this.parch = parch;
        this.ticket = ticket;
        this.fare = fare;
        this.cabin = cabin;
        this.embarked = embarked;
        this.homeDest = homeDest;
        this.boat = boat;
        this.body = body;
    }

    public Integer getPclass() {
        return pclass;
    }

    public void setPclass(Integer pclass) {
        this.pclass = pclass;
    }

    public Integer getSurvived() {
        return survived;
    }

    public void setSurvived(Integer survived) {
        this.survived = survived;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSex() {
        return sex;
    }

    public void setSex(String sex) {
        this.sex = sex;
    }

    public Double getAge() {
        return age;
    }

    public void setAge(Double age) {
        this.age = age;
    }

    public Integer getSibSp() {
        return sibSp;
    }

    public void setSibSp(Integer sibSp) {
        this.sibSp = sibSp;
    }

    public Integer getParch() {
        return parch;
    }

    public void setParch(Integer parch) {
        this.parch = parch;
    }

    public String getTicket() {
        return ticket;
    }

    public void setTicket(String ticket) {
        this.ticket = ticket;
    }

    public Double getFare() {
        return fare;
    }

    public void setFare(Double fare) {
        this.fare = fare;
    }

    public String getCabin() {
        return cabin;
    }

    public void setCabin(String cabin) {
        this.cabin = cabin;
    }

    public String getEmbarked() {
        return embarked;
    }

    public void setEmbarked(String embarked) {
        this.embarked = embarked;
    }

    public String getHomeDest() {
        return homeDest;
    }

    public void setHomeDest(String homeDest) {
        this.homeDest = homeDest;
    }

    public String getBoat() {
        return boat;
    }

    public void setBoat(String boat) {
        this.boat = boat;
    }

    public Integer getBody() {
        return body;
    }

    public void setBody(Integer body) {
        this.body = body;
    }

    @Override
    public String toString() {
        return "Passenger{" +
                "pclass=" + pclass +
                ", survived=" + survived +
                ", name='" + name + '\'' +
                ", sex='" + sex + '\'' +
                ", age=" + age +
                ", sibSp=" + sibSp +
                ", parch=" + parch +
                ", ticket='" + ticket + '\'' +
                ", fare=" + fare +
                ", cabin='" + cabin + '\'' +
                ", embarked='" + embarked + '\'' +
                ", homeDest='" + homeDest + '\'' +
                ", boat='" + boat + '\'' +
                ", body=" + body +
                '}'+"\n";
    }
}
